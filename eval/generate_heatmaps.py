import numpy as np
import torchvision.transforms.functional
from tqdm import tqdm
import os

import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, StackDataset
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from torchcam.methods import GradCAM, GradCAMpp, ScoreCAM
# from eval.cam.grad_cam import GradCAM
from torchvision.transforms.functional import normalize, resize, to_pil_image

from eval.utils import *
from models.image_model import ImageClassificationModel
from eval.evaluation import CausalMetric, auc, gkern
from training_utils import TEXT_CLASSES, DATASETS_TO_CLASSES
# from eval.cam import GradCAM, EigenGradCAM, ScoreCAM, GradCAMPlusPlus
from torchcam.utils import overlay_mask
from datasets.imagefolder_cgc_ssl import ImageFolder as CGCImageFolder
import argparse


parser = argparse.ArgumentParser(description="PyTorch Equivariance Evaluation")

parser.add_argument(
    "--cam", dest="cam", default='grad-cam', type=str, help="explainability method to use"
)
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
    "--training-method", dest="training_method", default="baseline", type=str, help="training scheme"
)
parser.add_argument(
    "--dataset", dest="dataset", default='imagenet-s50', type=str, help="dataset to evaluate on"
)
parser.add_argument(
    "--img-data", dest="img_data", default='data/ImageNetS50/val', type=str, help="path to images"
)
parser.add_argument(
    "--save-output", dest="save_output", default=None, type=str, help="path to save saliency maps"
)


class ImageFolderWithPaths(datasets.ImageFolder):

    def __init__(self, img_root: str, img_transforms, *args, **kwargs):
        super(ImageFolderWithPaths, self).__init__(img_root, img_transforms, *args, **kwargs)

    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]
        path = os.path.join(*(path.split('\\')[-2:]))
        path = path.replace('.JPEG', '.png')

        return (img, label, path)


def main():
    args = parser.parse_args()

    cudnn.benchmark = True
    #args.pretrained = True

    if not args.save_output:
        raise Exception("The save-output path must be provided")

    try:
        os.makedirs(os.path.join(args.save_output, 'saliency_maps'))
    except OSError as error:
        pass

    if args.pretrained:
        if args.arch == 'resnet18':
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif args.arch == 'resnet50':
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise NotImplementedError()
        target_layer = net.layer4[-1]
    else:
        net = ImageClassificationModel(
            arch=args.arch,
            num_classes=DATASETS_TO_CLASSES[args.dataset],
            pretrained=False,
            lr=None,
            momentum=None,
            weight_decay=None,
            total_steps=None)
        if not args.ckpt_path:
            raise Exception("Pretrained is set to False, but not checkpoint path found")
        state_dict = torch.load(args.ckpt_path)["state_dict"]
        if args.training_method == 'lico':
            state_dict = {k.replace("image_model.", ""): v for k, v in state_dict.items() if "image_model." in k}
        # load params
        net.load_state_dict(state_dict)

        target_layer = net._model.layer4[-1]

    net.cuda()
    cam = GradCAM(model=net, target_layer=target_layer)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #normalize,
    ])

    val_dataset = ImageFolderWithPaths(args.img_data, img_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=50)

    for img, label, paths in tqdm(val_dataloader):
        norm_img = normalize(img).cuda()
        output = net(norm_img)
        salience = cam(label.tolist(), output)[0]

        for idx in range(img.shape[0]):
            image, sal, path = img[idx], salience[idx], paths[idx]
            result = overlay_mask(to_pil_image(image), to_pil_image(sal.detach().cpu(), mode='F'), alpha=0.3)

            cls_folder, file_name = path.split(os.sep)[-2:]

            try:
                os.makedirs(os.path.join(args.save_output, 'saliency_maps', cls_folder))
            except OSError as error:
                pass

            cv2.imwrite(os.path.join(args.save_output, 'saliency_maps', cls_folder, file_name),
                        np.array(result)[:, :, ::-1])

            # cv2.imshow("saliency", np.array(result)[:, :, ::-1])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
