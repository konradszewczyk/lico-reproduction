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
from torchvision.transforms.functional import normalize, resize, to_pil_image

from eval.utils import *
from models.LICO_model import LICOModel
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
    "--img_data", dest="img_data", default='data/ImageNetS50/val', type=str, help="path to images"
)
parser.add_argument(
    "--output", dest="output", default=None, type=str, help="path to save saliency maps"
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
    cam = GradCAMpp(model=net, target_layer=target_layer)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # normalize,
    ])

    val_dataset = ImageFolderWithPaths(args.img_data, img_transforms)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=8)

    for img, label, paths in val_dataloader:
        norm_img = normalize(img).cuda()
        output = net(norm_img)
        salience = cam(label.tolist(), output)[0]

        #salience = [to_pil_image(sal.unsqueeze(0)).resize(img.shape[2:], resample=Image.BICUBIC) for sal in salience]
        #salience = np.stack([np.float32(sal) / 255 for sal in salience], axis=0)
        #salience = torch.tensor(salience)

        for idx in range(img.shape[0]):
            image, sal, path = img[idx], salience[idx], paths[idx]
            result = overlay_mask(to_pil_image(image), to_pil_image(sal.detach().cpu(), mode='F'), alpha=0.3)

            # Convert both original image and result to numpy arrays
            original_image_np = np.array(to_pil_image(image))
            result_np = np.array(result)

            # Stack the original image and the result side by side
            combined = np.hstack((original_image_np, result_np))

            cv2.imshow(f"Original + Saliency for {TEXT_CLASSES[args.dataset][label.tolist()[idx]]}", combined[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # idx = np.argmax(label > 1)

        # sal_img = cv2.applyColorMap((salience[idx, :, :] * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        # weight = cv2.addWeighted(img, 0.5, sal_img, 0.7, 5)

        #sal_img = cv2.applyColorMap((salience[0][idx].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        #weight = cv2.addWeighted(img, 0.5, sal_img, 0.7, 5)

        #result = overlay_mask(to_pil_image(img), to_pil_image(salience[0][idx].detach().cpu().squeeze(0), mode='F'), alpha=0.3)
        #cv2.imshow("saliency", salience[idx, :, :, np.newaxis] * img)
        # cv2.imshow("saliency", np.array(result)[:, :, ::-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #cv2.imshow("saliency", salience[idx, :, :, np.newaxis] * img)
        # cv2.imshow("saliency", weight)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    results_df = pd.DataFrame({'label': res_labels, 'segmentation_score': res_scores})
    results_df['label'] = results_df['label'].apply(lambda x: TEXT_CLASSES[args.dataset][int(x)])

    print(results_df.groupby('label').mean())
    print("============================")
    print("Validation segmentation score:", results_df['segmentation_score'].mean())


if __name__ == "__main__":
    main()
