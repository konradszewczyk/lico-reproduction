import argparse
import os
import random
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from models.LICO_model import LICOModel, tokenize_targets
from models.image_model import ImageClassificationModel
from training_utils import get_logger, DATASETS_TO_CLASSES, TEXT_CLASSES


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--training-method', type=str, default='baseline',
                    choices=['baseline', 'LICO'])
parser.add_argument('--alpha', type=float, default=10., help='alpha for LICO loss')
parser.add_argument('--beta', type=float, default=1., help='beta for LICO loss')

parser.add_argument('--context_tokens', type=int, default=12, help='number of learnable text tokens')
parser.add_argument('--learnable_context', type=bool, default=True, help='whether to train params of context tokens')
parser.add_argument('--enable_cls_prompts', default=False, action=argparse.BooleanOptionalAction, help='enable trainable prompts per class')
parser.add_argument('--dynamic_context', type=bool, default=True, help='whether to shuffle trainable context tokens')
parser.add_argument('--context_position', type=str, default='end', help='part of the prompts where the context tokens should be inserted')

parser.add_argument('--data', metavar='DIR', default='data',
                    help='path to dataset')
parser.add_argument('--train_mm_temp', type=bool, default=True, help='whether to train the MM temperature parameter')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--save_dir', default='checkpoint', type=str, metavar='SV_PATH',
                    help='path to save checkpoints (default: none)')
parser.add_argument('--log_dir', default='logs', type=str, metavar='LG_PATH',
                    help='path to write logs (default: logs)')
parser.add_argument('--dataset', type=str, default='cifar100',
                            help='dataset to use: [cifar100, imagenet, cub, aircraft, flowers, cars]')


def main():
    # fp16 precision for speed
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()

    # args.seed = 1
    # args.training_method = 'LICO'
    # args.epochs = 10
    # args.batch_size = 64
    # args.enable_cls_prompts = False
    #
    # # args.data = 'C:/Users/Mikhail/Datasets/ImagenetS/ImageNetS50'
    # # args.dataset = 'imagenet-s50'
    # args.data = 'C:/Users/Mikhail/Datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC'
    # args.dataset = 'imagenet'
    # args.workers = 8
    # args.arch = 'resnet18'
    # args.pretrained = True

    # args.resume = 'checkpoint/LICO-cifar100-resnet18-seed_42epoch=1-train_loss=4.00-val_loss=3.81-val_acc1=0.15.ckpt'
    # args.evaluate = True

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(logpath=os.path.join(args.log_dir, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    if args.seed is not None:
        pl.seed_everything(args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.evaluate:
        assert args.resume, 'Cannot evaluate without a checkpoint'
        assert os.path.isfile(args.resume), 'Checkpoint not found'
        evaluate(args)
    else:
        train(args)


def create_dataloaders(args):
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    if args.dataset == 'cifar100':
        if not os.path.exists(os.path.join(args.data, 'train')):
            from download_datasets import download_and_prepare_cifar100
            download_and_prepare_cifar100(args.data)
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        # print('batch size overwritten to 64')
        # args.batch_size = 64
        # cifar100 has only 2 sets of data
        testdir = os.path.join(args.data, 'val')

    elif args.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # print('batch size overwritten to 128')
        # args.batch_size = 128
        testdir = os.path.join(args.data, 'val')

    elif args.dataset == 'imagenet-s50':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # print('batch size set to 128')
        # args.batch_size = 128
        testdir = os.path.join(args.data, 'val')

    else:
        raise NotImplementedError

    common_args = {
        'batch_size': args.batch_size,
        'num_workers': args.workers,
        'pin_memory': True,
        'persistent_workers': True
    }

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **common_args)

    val_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    val_dataset = datasets.ImageFolder(valdir, val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **common_args)

    test_dataset = datasets.ImageFolder(testdir, val_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **common_args)
    return train_loader, val_loader, test_loader


def make_model(args, total_steps):
    num_classes = DATASETS_TO_CLASSES[args.dataset]

    if args.training_method == 'baseline':
        if args.resume:
            print("\nLoading checkpoint '{}'\n".format(args.resume))
            model = ImageClassificationModel.load_from_checkpoint(args.resume)
        else:
            model = ImageClassificationModel(
                pretrained=args.pretrained, arch=args.arch, lr=args.lr,
                momentum=args.momentum, weight_decay=args.weight_decay,
                num_classes=num_classes, total_steps=total_steps
            )
    elif args.training_method == 'LICO':
        target_names = tokenize_targets(TEXT_CLASSES[args.dataset])
        if args.resume:
            print("\nLoading checkpoint '{}'\n".format(args.resume))
            model = LICOModel.load_from_checkpoint(
                # strict=False because text model is not in the checkpoint
                args.resume, strict=False,
            )
        else:
            image_model = ImageClassificationModel(
                pretrained=args.pretrained, arch=args.arch, lr=args.lr,
                momentum=args.momentum, weight_decay=args.weight_decay, num_classes=num_classes,
                total_steps=total_steps
            )
            model = LICOModel(image_model, target_names=target_names,
                              alpha=args.alpha, beta=args.beta, context_tokens=args.context_tokens,
                              learnable_context=args.learnable_context, dynamic_context=args.dynamic_context,
                              train_mm_temp=args.train_mm_temp, enable_cls_prompts=args.enable_cls_prompts, num_classes=num_classes)
    else:
        raise NotImplementedError
    return model


def train(args):
    print(f"Starting training with the following configuration:\n"
          f" - Dataset: {args.dataset} (Path: {args.data})\n"
          f" - Architecture: {args.arch}\n"
          f" - Training Method: {args.training_method}\n"
          f" - Seed: {args.seed}\n"
          + (f" - Alpha: {args.alpha}\n - Beta: {args.beta}\n" if (args.training_method == 'LICO') else "")
          + (f" - Resuming from checkpoint: {args.resume}" if args.resume else ""))
    cudnn.benchmark = True

    train_loader, val_loader, _ = create_dataloaders(args)

    total_steps = len(train_loader) * args.epochs

    model = make_model(args, total_steps)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=f'{args.training_method}-{args.dataset}-{args.arch}-'
                 f'{("seed_" + str(args.seed) + "-") if args.seed else ""}' +
                 '{epoch}-{train_loss:.2f}-{val_loss:.2f}-{val_acc1:.2f}',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=True,
        logger=WandbLogger(project="lico-reproduction", config=args, log_model=True),
        gradient_clip_val=0.5,
    )

    trainer.fit(model, train_loader, val_loader)


def evaluate(args):
    print("Evaluating the model")
    _, _, test_loader = create_dataloaders(args)

    model = make_model(args, 1)

    trainer = pl.Trainer(
        enable_progress_bar=True,
        logger=WandbLogger(project="lico-reproduction-eval", config=args),
    )

    trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
