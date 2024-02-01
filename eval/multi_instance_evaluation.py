import numpy as np
import torchvision.transforms.functional
from tqdm import tqdm
import os

import pandas as pd
import cv2

import xml.etree.ElementTree as ET

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
    "--annotation-data", dest="annotation_data", default='data/ImageNetS50/validation-segmentation', type=str,
    help="path to bounding box annotations"
)
parser.add_argument(
    "--save-output", dest="save_output", default=None, type=str, help="folder where to save the output DataFrame"
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
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    res_labels = np.array([])
    res_scores = np.array([])

    for img, label, paths in val_dataloader:
        norm_img = normalize(img).cuda()
        output = net(norm_img)
        salience = cam(label.tolist(), output)[0]

        salience = [to_pil_image(sal.unsqueeze(0)).resize(img.shape[2:], resample=Image.BICUBIC) for sal in salience]
        salience = np.stack([np.float32(sal) / 255 for sal in salience], axis=0)
        salience = torch.tensor(salience)

        for idx in range(img.shape[0]):
            image, sal, path = img[idx], salience[idx], paths[idx]

            cls_folder, file_name = path.split(os.sep)

            #xml_doc = ET.parse(os.path.join(args.annotation_data, cls_folder, file_name))
            xml_doc = ET.parse('data/ImageNetS50/Annotations/Annotations/CLS-LOC/train/n01440764/n01440764_18.xml')
            xml_root = xml_doc.getroot()
            object_instances = xml_root.findall('object')
            instance_no = len(object_instances)
            #if instance_no < 2:
            #    continue

            width = int(xml_root.find('size').find('width').text)
            height = int(xml_root.find('size').find('height').text)

            for instance in object_instances:
                bndbox = instance.find('bndbox')
                x_min = int(int(bndbox.find('xmin').text) / (width / image.shape[2]))
                x_max = int(int(bndbox.find('xmax').text) / (width / image.shape[2]))
                y_min = int(int(bndbox.find('ymin').text) / (height / image.shape[1]))
                y_max = int(int(bndbox.find('ymax').text) / (height / image.shape[1]))

                print(x_min, x_max, y_min, y_max)

                bbox_mask = torch.zeros((image.shape[1], image.shape[2]))
                bbox_mask[y_min:y_max, x_min:x_max] = 1

                total_salience = sal.sum()
                norm_salience = sal / (total_salience + 1e-5)

                instance_percentage = norm_salience * bbox_mask
                print(instance_percentage.sum())

            raise NotImplementedError()

        total_salience = salience.sum(dim=(1, 2), keepdims=True)
        norm_salience = salience / (total_salience + 1e-5)

        segmentation_score = norm_salience * seg
        segmentation_score = segmentation_score.sum(dim=(1, 2))

        res_labels = np.concatenate([res_labels, label.numpy()])
        res_scores = np.concatenate([res_scores, segmentation_score.numpy()])


    results_df = pd.DataFrame({'label': res_labels, 'segmentation_score': res_scores})
    results_df['label'] = results_df['label'].apply(lambda x: TEXT_CLASSES[args.dataset][int(x)])

    #print(results_df.groupby('label').mean())
    grouped_results = results_df.groupby('label').aggregate({'segmentation_score': ['count', 'mean']})
    grouped_results = grouped_results.droplevel(0, axis=1)
    print(grouped_results)
    if args.save_output:
        try:
            os.makedirs(args.save_output)
        except OSError as error:
            pass
        grouped_results.to_csv(os.path.join(args.save_output, 'content_heatmap_cls.csv'))
    print("============================")
    total_results = results_df.aggregate({'segmentation_score': ['count', 'mean']})
    print("Validation segmentation score:", total_results)
    if args.save_output:
        total_results.to_csv(os.path.join(args.save_output, 'content_heatmap.csv'))


if __name__ == "__main__":
    main()