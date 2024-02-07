## Code to evaluate the pre-trained model and our CGC trained model with Insertion AUC score.

import argparse
import json
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from matplotlib import pyplot as plt
from tqdm import tqdm

from eval.cam import GradCAM
from datasets.imagefolder_cgc_ssl import ImageFolder as CGCImageFolder
from models.image_model import ImageClassificationModel
from training_utils import DATASETS_TO_CLASSES


def salience_equivariance_score(args):
    cudnn.benchmark = True
    n_classes = DATASETS_TO_CLASSES[args.dataset]

    arch = "resnet18"

    if args.pretrained:
        net = models.resnet18(pretrained=True)
        # Using the last layer of the 4th block
        target_layer = net.layer4[-1]
        print("Using pretrained model")
    else:
        if not args.ckpt_path:
            raise Exception("Pretrained is set to False, but no checkpoint path found")
        if "cgc" in args.ckpt_path:
            print(f"Using model checkpoint: {args.ckpt_path}")

            import models.resnet_multigpu_cgc as resnet

            net = resnet.resnet18()
            net.fc = nn.Linear(512, n_classes)
            state_dict = torch.load(args.ckpt_path)["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            net.load_state_dict(state_dict)
            target_layer = net.layer4[-1]
        else:
            print(f"Using model checkpoint: {args.ckpt_path}")

            state_dict = torch.load(args.ckpt_path)["state_dict"]
            net = ImageClassificationModel(
                arch=arch,
                pretrained=False,
                lr=None,
                num_classes=n_classes,
                momentum=None,
                weight_decay=None,
                total_steps=10000,  # a placeholder value.
            )
            # Remove image_model prefix from loaded model. It is probably generated based on the
            # module folder name.
            state_dict = {
                k.replace("image_model.", ""): v for k, v in state_dict.items()
            }
            # Remove LICO parameters. Those were supposed to be thrown away after the training.
            filer = (
                "learnable_prompts",
                "target_names",
                "projection",
                "criterion.mm_loss.temperature",
            )
            state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith(filer)
            }
            net.load_state_dict(state_dict)
            # Using the last layer of the 4th block
            target_layer = net._model.layer4[-1]

    run_dir = os.path.join("consistency-output", args.save_dir)

    # device = "cuda:0"
    device = "cpu"
    net = net.to(device)

    os.makedirs(run_dir, exist_ok=True)

    cam = GradCAM(model=net, target_layer=target_layer, use_cuda=True)

    consis_cosim = get_consistency_per_data_subset(
        net, cam, run_dir, device, args.dataset
    )
    results = {
        "mean-cossim": consis_cosim,
    }

    # Write the consistencies list to a JSON file
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, sort_keys=True, indent=4)

    print("Final:\n Consistency - {:.5f}".format(consis_cosim))


def get_consistency_per_data_subset(net, cam, save_dir, device, dataset):
    batch_size = 50

    viz = True

    data_loader = torch.utils.data.DataLoader(
        dataset=CGCImageFolder(
            f"./data/{dataset}/val/"
        ),  # transforms are handled within the implementation
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    net = net.eval()

    similarities = []
    import torch.nn.functional as F

    for j, data_loader_sample in enumerate(
        tqdm(data_loader, total=len(data_loader), desc="Interpreting images")
    ):
        # No need for the xe images
        (
            _,
            images,
            aug_images,
            transforms_i,
            transforms_j,
            transforms_h,
            transforms_w,
            hor_flip,
            targets,
        ) = [samp.to(device) for samp in data_loader_sample]

        b_size_real = images.shape[0]

        # Get saliency maps for images and augmented images
        img_gcam_maps_batch = cam(input_tensor=images, target_category=targets)
        aug_img_gcam_maps_batch = cam(input_tensor=aug_images, target_category=targets)

        img_gcam_maps_batch = torch.from_numpy(img_gcam_maps_batch).to(device)
        aug_img_gcam_maps_batch = torch.from_numpy(aug_img_gcam_maps_batch).to(device)

        # Augmentation parameters
        aug_params_dict = {
            "transforms_i": transforms_i,
            "transforms_j": transforms_j,
            "transforms_h": transforms_h,
            "transforms_w": transforms_w,
            "hor_flip": hor_flip,
        }

        # transform the saliency maps of non-augmented images
        img_gcam_maps_aug_batch = apply_transforms_to_heatmaps(
            img_gcam_maps_batch, aug_params_dict, device
        )

        if viz:
            viz_grid(
                images[:10],
                img_gcam_maps_batch[:10],
                img_gcam_maps_aug_batch[:10],
                aug_images[:10],
                aug_img_gcam_maps_batch[:10],
                targets=targets[:10],
                dir=save_dir,
                idx=j,
            )

        # Flatten and compute similarity
        img_gcam_maps_aug_batch = img_gcam_maps_aug_batch.view(b_size_real, -1)
        aug_img_gcam_maps_batch = aug_img_gcam_maps_batch.view(b_size_real, -1)

        sim = F.cosine_similarity(img_gcam_maps_aug_batch, aug_img_gcam_maps_batch)
        similarities.append(sim.mean().item())

    return torch.tensor(similarities).mean().item()


def viz_grid(*batches, targets, dir, idx):
    rows = max(len(batch) for batch in batches)
    columns = len(batches)  # Number of batches
    fig = plt.figure(figsize=(10, 10))

    for i in range(rows):
        for j, batch in enumerate(batches):
            if i < len(batch):
                image = batch[i]
                # Convert the tensor to numpy array
                image_np = image.cpu().numpy()

                # Transpose the numpy array if necessary
                if (
                    image_np.shape[0] == 3
                ):  # Check if the image tensor is in the format (channels, height, width)
                    # Scale the values to [0, 1] range
                    image_np = (image_np - image_np.min()) / (
                        image_np.max() - image_np.min()
                    )
                    image_np = np.transpose(
                        image_np, (1, 2, 0)
                    )  # Transpose to (height, width, channels)

                # Display the image
                ax = fig.add_subplot(rows, columns, i * columns + j + 1)
                # Hide X and Y axes label marks
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.yaxis.set_tick_params(labelleft=False)
                # Hide X and Y axes tick marks
                ax.set_xticks([])
                ax.set_yticks([])
                # Show as image
                ax.imshow(image_np)

    fig.suptitle(f"CIFAR-100 val set class {targets.tolist()}")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join(dir, f"{idx}.png"))
    # plt.show()
    plt.close()


def apply_transforms_to_heatmaps(heatmaps, aug_params_dict, device):
    orig_gradcam_mask = heatmaps
    transforms_i = aug_params_dict["transforms_i"]
    transforms_j = aug_params_dict["transforms_j"]
    transforms_h = aug_params_dict["transforms_h"]
    transforms_w = aug_params_dict["transforms_w"]
    hor_flip = aug_params_dict["hor_flip"]
    gpu_batch_len = transforms_i.shape[0]
    augmented_orig_gradcam_mask = torch.zeros_like(orig_gradcam_mask).to(device)
    for b in range(gpu_batch_len):
        # convert orig_gradcam_mask to image
        orig_gcam = orig_gradcam_mask[b]
        orig_gcam = orig_gcam[
            transforms_i[b] : transforms_i[b] + transforms_h[b],
            transforms_j[b] : transforms_j[b] + transforms_w[b],
        ]
        # We use torch functional to resize without breaking the graph
        orig_gcam = orig_gcam.unsqueeze(0).unsqueeze(0)
        orig_gcam = F.interpolate(orig_gcam, size=224, mode="bilinear")
        orig_gcam = orig_gcam.squeeze()
        if hor_flip[b]:
            orig_gcam = orig_gcam.flip(-1)
        augmented_orig_gradcam_mask[b, :, :] = orig_gcam[:, :]
    return augmented_orig_gradcam_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Equivariance Evaluation")
    parser.add_argument(
        "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
    )
    parser.add_argument(
        "--ckpt-path", dest="ckpt_path", type=str, help="path to checkpoint file"
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, help="dataset name", default="cifar100"
    )
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=str,
        help="path to save dir",
        default="pretrained",
    )

    args = parser.parse_args()

    salience_equivariance_score(args)
