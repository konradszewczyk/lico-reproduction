## Code to evaluate the pre-trained model and our CGC trained model with Insertion AUC score.

import numpy as np
from tqdm import tqdm

import torch
import os
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from utils import *
from evaluation import CausalMetric, auc, gkern
from eval.cam.grad_cam import GradCAM
from download_datasets import download_and_prepare_cifar100
from models.image_model import ImageClassificationModel
from training_utils import get_logger, DATASETS_TO_CLASSES
import argparse

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch AUC Metric Evaluation")
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument(
    "--ckpt-path", dest="ckpt_path", type=str, help="path to checkpoint file"
)
parser.add_argument("--data", metavar="DIR", default="data", help="path to dataset")
parser.add_argument(
    "--workers",
    default=4,
    type=int,
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--batch-size", dest="batch_size", default=64, type=int, help="size of batch"
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar100",
    help="dataset to use: [cifar100, imagenet, imagenet-s50]",
)
parser.add_argument(
    "--method",
    type=str,
    default="ins",
    help="method to use: [ins, del, both]",
)


def main():
    args = parser.parse_args()

    n_classes = DATASETS_TO_CLASSES[args.dataset]
    state_dict = torch.load(args.ckpt_path)["state_dict"]
    net = ImageClassificationModel(
        arch=args.arch,
        pretrained=False,
        lr=None,
        num_classes=n_classes,
        momentum=None,
        weight_decay=None,
        total_steps=10000,  # a placeholder value.
    )
    # Remove image_model prefix from loaded model. It is probably generated based on the
    # module folder name.
    state_dict = {k.replace("image_model.", ""): v for k, v in state_dict.items()}
    # Remove LICO parameters. Those were supposed to be thrown away after the training.
    filer = (
        "learnable_prompts",
        "target_names",
        "projection",
        "criterion.mm_loss.temperature",
    )
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(filer)}
    net.load_state_dict(state_dict)

    target_layer = net._model.layer4
    cam = GradCAM(model=net, target_layer=target_layer, use_cuda=True)

    scores = {"del": [], "ins": []}
    for val_dataload in create_val_dataloaders(
        args.data,
        args.dataset,
        args.batch_size,
        args.workers,
    ):
        ins_auc_score, del_auc_score = get_auc_per_data_subset(
            net, cam, val_dataload, n_classes, args.method
        )
        scores["ins"].append(ins_auc_score)
        scores["del"].append(del_auc_score)

    print("----------------------------------------------------------------")
    print("Final:\nInsertion - {:.5f}".format(np.mean(scores["ins"])))
    print("Final:\nDeletion - {:.5f}".format(np.mean(scores["del"])))


def create_val_dataloaders(data_dir, dataset, batch_size, n_workers):
    # Data loading code
    if dataset == "cifar100":
        download_and_prepare_cifar100(data_dir)
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )
        sample_size = 1000
    elif dataset == "imagenet":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        sample_size = 1000
    elif dataset == "imagenet-s50":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        sample_size = 1000
    else:
        raise NotImplementedError

    val_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    valdir = os.path.join(data_dir, "val")
    val_dataset = datasets.ImageFolder(valdir, val_transforms)
    val_size = len(val_dataset)
    for i in range(int(np.ceil(val_size / sample_size))):
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=n_workers,
            # We only load N samples in the memory at the time
            # to prevent OOM error.
            sampler=RangeSampler(
                range(sample_size * i, min(sample_size * (i + 1), val_size))
            ),
        )
        yield val_loader


def get_auc_per_data_subset(net, cam, data_loader, n_classes, method):
    net = net.train()
    images = []
    targets = []
    gcam_exp = []

    batch_size = len(data_loader)
    for j, (img, trg) in enumerate(
        tqdm(data_loader, total=batch_size, desc="Loading images")
    ):
        grayscale_gradcam = cam(input_tensor=img, target_category=trg)
        for k in range(batch_size):
            images.append(img[k])
            targets.append(trg[k])
            gcam_exp.append(grayscale_gradcam[k])

    images = torch.stack(images).cpu().numpy()
    gcam_exp = np.stack(gcam_exp)
    images = np.asarray(images)
    gcam_exp = np.asarray(gcam_exp)

    images = images.reshape((-1, 3, 224, 224))
    gcam_exp = gcam_exp.reshape((-1, 224, 224))
    print("Finished obtaining CAM")

    model = nn.Sequential(net, nn.Softmax(dim=1))
    model = model.eval()
    model = model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    # To use multiple GPUs
    ddp_model = nn.DataParallel(model)

    # we use blur as the substrate function
    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)
    # Function that blurs input image
    blur = lambda x: F.conv2d(x, kern, padding=klen // 2)
    # This is what RISE codebase uses for graying out the image.
    gray = torch.zeros_like

    insertion = CausalMetric(ddp_model, "ins", 224 * 8, substrate_fn=blur)
    deletion = CausalMetric(ddp_model, "del", 224 * 8, substrate_fn=gray)

    # Example for flotting insertion and deletion for one image.
    # img = torch.unsqueeze(torch.from_numpy(images.astype("float32"))[0], 0)
    # insertion.single_run(
    #     img,
    #     gcam_exp[0],
    #     verbose=2, # generated plot for all insertion steps
    #     save_to="<save-path>",
    # )
    # deletion.single_run(
    #     img,
    #     gcam_exp[0],
    #     verbose=2, # generated plot for all deletionk steps
    #     save_to="<save-path>",
    # )

    # Evaluate insertion and deletion score
    auc_ins, auc_dels = 0, 0
    if method == "ins" or method == "both":
        h_ins = insertion.evaluate(
            torch.from_numpy(images.astype("float32")),
            gcam_exp,
            batch_size,
            n_classes=n_classes,
        )
        auc_ins = auc(h_ins.mean(1))
    if method == "del" or method == "both":
        h_del = deletion.evaluate(
            torch.from_numpy(images.astype("float32")),
            gcam_exp,
            batch_size,
            n_classes=n_classes,
        )
        auc_dels = auc(h_del.mean(1))

    model = model.train()
    for p in model.parameters():
        p.requires_grad = True

    return auc_ins, auc_dels


if __name__ == "__main__":
    main()
