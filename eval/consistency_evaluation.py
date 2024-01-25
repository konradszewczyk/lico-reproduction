## Code to evaluate the pre-trained model and our CGC trained model with Insertion AUC score.

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from eval.utils import *
from eval.evaluation import CausalMetric, auc, gkern
from eval.cam import GradCAM
from datasets.imagefolder_cgc_ssl import ImageFolder as CGCImageFolder
import argparse

parser = argparse.ArgumentParser(description="PyTorch Equivariance Evaluation")
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument(
    "--ckpt-path", dest="ckpt_path", type=str, help="path to checkpoint file"
)


def main():
    args = parser.parse_args()

    cudnn.benchmark = True

    if args.pretrained:
        net = models.resnet18(pretrained=True)
    else:
        net = models.resnet18()
        if not args.ckpt_path:
          raise Exception("Pretrained is set to False, but not checkpoint path found")
        state_dict = torch.load(args.ckpt_path)["state_dict"]

        # remove the module prefix if model was saved with DataParallel
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # load params
        net.load_state_dict(state_dict)

    # transform_list = [
    #     transforms.Resize((256, 256)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # ]
    
    # Last layer?
    target_layer = net.layer4[-1]
    cam = GradCAM(model=net, target_layer=target_layer, use_cuda=True)

    consistencies = []
    
    # we process the images in 10 set of 1k each and compute mean
    # num_subsets = 10
    num_subsets = 1
    for i in range(num_subsets):
        consistency = get_consistency_per_data_subset(i, net, cam)
        
        consistencies.append(consistency.item())
        print("Finished evaluating the consistency metrics...")

    print("----------------------------------------------------------------")
    print("Final:\n Consistency - {:.5f}".format(np.mean(consistencies)))


def apply_transforms_to_heatmaps(heatmaps, aug_params_dict):
    orig_gradcam_mask = heatmaps
    transforms_i = aug_params_dict['transforms_i']
    transforms_j = aug_params_dict['transforms_j']
    transforms_h = aug_params_dict['transforms_h']
    transforms_w = aug_params_dict['transforms_w']
    hor_flip = aug_params_dict['hor_flip']
    gpu_batch_len = transforms_i.shape[0]
    augmented_orig_gradcam_mask = torch.zeros_like(orig_gradcam_mask)
    for b in range(gpu_batch_len):
        # convert orig_gradcam_mask to image
        orig_gcam = orig_gradcam_mask[b]
        orig_gcam = orig_gcam[transforms_i[b]: transforms_i[b] + transforms_h[b],
                    transforms_j[b]: transforms_j[b] + transforms_w[b]]
        # We use torch functional to resize without breaking the graph
        orig_gcam = orig_gcam.unsqueeze(0).unsqueeze(0)
        orig_gcam = F.interpolate(orig_gcam, size=224, mode='bilinear')
        orig_gcam = orig_gcam.squeeze()
        if hor_flip[b]:
            orig_gcam = orig_gcam.flip(-1)
        augmented_orig_gradcam_mask[b, :, :] = orig_gcam[:, :]
    return augmented_orig_gradcam_mask


def get_consistency_per_data_subset(range_index, net, cam):
    batch_size = 10
    
    subset_size = 1000
    data_loader = torch.utils.data.DataLoader(
        # dataset=datasets.ImageFolder("./data/val/", preprocess),
        dataset=CGCImageFolder("./data/val/"),   # transforms are handled within the implementation
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=RangeSampler(range(subset_size * range_index, subset_size * (range_index + 1))),
    )

    net = net.train()

    similarities = []
    import torch.nn.functional as F

    for j, data_loader_sample in enumerate(
        tqdm(data_loader, total=len(data_loader), desc="Interpreting images")
    ):
        # No need for the xe images
        _,  images, aug_images, transforms_i, transforms_j, transforms_h, transforms_w, hor_flip, targets = data_loader_sample
        
        # Get saliency maps for images and augmented images
        img_gcam_maps_batch = cam(input_tensor=images, target_category=targets)
        aug_img_gcam_maps_batch = cam(input_tensor=aug_images, target_category=targets)
        
        img_gcam_maps_batch = torch.from_numpy(img_gcam_maps_batch)
        aug_img_gcam_maps_batch = torch.from_numpy(aug_img_gcam_maps_batch)
        
        # Augmentation parameters
        aug_params_dict = {'transforms_i': transforms_i, 'transforms_j': transforms_j, 'transforms_h': transforms_h, 'transforms_w': transforms_w, 'hor_flip': hor_flip}
        
        # transform the saliency maps of non-augmented images
        img_gcam_maps_aug_batch = apply_transforms_to_heatmaps(img_gcam_maps_batch, aug_params_dict)
        
        # Flatten and compute similarity
        img_gcam_maps_aug_batch = img_gcam_maps_aug_batch.view(batch_size, -1)
        aug_img_gcam_maps_batch = aug_img_gcam_maps_batch.view(batch_size, -1)
        
        sim = F.cosine_similarity(img_gcam_maps_aug_batch, aug_img_gcam_maps_batch)
        similarities.append(sim.mean())
    
    return torch.tensor(similarities).mean()


def viz_grid(*batches):
        rows = max(len(batch) for batch in batches)
        columns = len(batches)  # Number of batches
        fig = plt.figure(figsize=(10, 10))

        for i in range(rows):
            for j, batch in enumerate(batches):
                if i < len(batch):
                    image = batch[i]
                    # Convert the tensor to numpy array
                    image_np = image.numpy()

                    # Scale the values to [0, 1] range
                    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

                    # Transpose the numpy array if necessary
                    if image_np.shape[0] == 3:  # Check if the image tensor is in the format (channels, height, width)
                        image_np = np.transpose(image_np, (1, 2, 0))  # Transpose to (height, width, channels)

                    # Display the image
                    ax = fig.add_subplot(rows, columns, i * columns + j + 1)
                    ax.imshow(image_np)
                    ax.set_title(f'Dimensions: {image_np.shape}')

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()


if __name__ == "__main__":
    main()
