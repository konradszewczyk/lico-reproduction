import numpy as np
from torch.utils.data.dataset import T_co
from tqdm import tqdm
import os

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, StackDataset
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from eval.utils import *
from models.image_model import ImageClassificationModel
from eval.evaluation import CausalMetric, auc, gkern
from training_utils import TEXT_CLASSES
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
parser.add_argument(
    "--arch", dest="arch", default='resnet18', type=str, help="model architecture"
)
parser.add_argument(
    "--img_data", dest="img_data", default='data/ImageNetS50/val', type=str, help="path to images"
)
parser.add_argument(
    "--seg_data", dest="seg_data", default='data/ImageNetS50/validation-segmentation', type=str, help="path to segmentation"
)



class ImageFolderWithSegmentation(datasets.ImageFolder):

    def __init__(self, img_root: str, seg_root: str, img_transforms, seg_transforms, *args, **kwargs):
        super(ImageFolderWithSegmentation, self).__init__(img_root, img_transforms, *args, **kwargs)
        self.seg_root = seg_root
        self.seg_transforms = seg_transforms

    def __getitem__(self, index):
        img, label = super(ImageFolderWithSegmentation, self).__getitem__(index)

        path = self.imgs[index][0]
        seg_name = os.path.join(self.seg_root, *(path.split('\\')[-2:]))
        seg_name = seg_name.replace('.JPEG', '.png')

        seg = Image.open(seg_name)
        seg = self.seg_transforms(seg).to(torch.int32)
        seg = seg[0] + 256 * seg[1] + 256**2 * seg[2]

        return (img, label, seg)


def main():
    args = parser.parse_args()

    cudnn.benchmark = True

    if args.pretrained:
        net = ImageClassificationModel(
            arch=args.arch,
            num_classes=50,
            pretrained=True,
            lr=None,
            momentum=None,
            weight_decay=None,
            total_steps=None)
        pass
    else:
        net = ImageClassificationModel(
            arch=args.arch,
            num_classes=50,
            pretrained=False,
            lr=None,
            momentum=None,
            weight_decay=None,
            total_steps=None)
        if not args.ckpt_path:
            raise Exception("Pretrained is set to False, but not checkpoint path found")
        state_dict = torch.load(args.ckpt_path)["state_dict"]

        # load params
        net.load_state_dict(state_dict)

    target_layer = net._model.layer4[-1]
    cam = GradCAM(model=net, target_layer=target_layer, use_cuda=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    seg_transforms = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(224),
        transforms.PILToTensor()
    ])

    val_dataset = ImageFolderWithSegmentation(args.img_data, args.seg_data, img_transforms, seg_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=128)

    res_labels = np.array([])
    res_scores = np.array([])

    for img, label, seg in val_dataloader:
        salience = cam(img, label)

        seg_label = label + 1
        seg_label = seg_label.view(-1, 1, 1)
        seg_binary = seg == seg_label
        seg_binary = seg_binary.numpy()

        total_salience = salience.sum(axis=(1, 2), keepdims=True)
        norm_salience = salience / total_salience

        segmentation_score = norm_salience * seg_binary
        segmentation_score = segmentation_score.sum(axis=(1, 2))

        res_labels = np.concatenate([res_labels, label])
        res_scores = np.concatenate([res_scores, segmentation_score])

    results_df = pd.DataFrame({'label': res_labels, 'segmentation_score': res_scores})
    results_df['label'] = results_df['label'].apply(lambda x: TEXT_CLASSES['imagenet-s50'][int(x)])

    print(results_df.groupby('label').mean())
    print("============================")
    print("Validation segmentation score:", results_df['segmentation_score'].mean())


if __name__ == "__main__":
    main()