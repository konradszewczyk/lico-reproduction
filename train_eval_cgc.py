## Code to train and evaluate a ResNet model with our CGC method on ImageNet/CUB-200/Cars-196/FGVC-Aircraft/VGG Flowers-102 datasets.
## This code is adapated from https://github.com/pytorch/examples/blob/master/imagenet/main.py

import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datasets.imagefolder_cgc_ssl import ImageFolder
from models.cosine_lr_scheduler import CosineLRScheduler
import models.resnet_multigpu_cgc as resnet
import wandb

import logging

from training_utils import get_logger, accuracy

model_names = ['resnet18' , 'resnet50']
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
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
parser.add_argument('--dataset', type=str, default='imagenet',
                            help='dataset to use: [imagenet, tiny_imagenet]')

parser.add_argument('-t', type=float, default=0.01)
parser.add_argument('--lambda', default=1000, type=float,
                    metavar='LAM', help='lambda hyperparameter for GCAM loss', dest='lambda_val')

best_acc1 = 0
global_step = 0


def main():
    torch.set_float32_matmul_precision('medium')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(logpath=os.path.join(args.log_dir, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, logger)


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self , T=0.01):
        super(NCESoftmaxLoss, self).__init__()
        self.T = T
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        x = torch.div(x, self.T)
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


def main_worker(gpu, ngpus_per_node, args, logger):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    kwargs = {}
    num_classes = 1000
    val_dir_name = 'val'
    if args.dataset == 'tiny_imagenet':
        kwargs = {'num_classes': 200}
        num_classes = 200
    elif args.dataset == 'cub':
        kwargs = {'num_classes': 200}
        num_classes = 200
        val_dir_name = 'test'
    elif args.dataset == 'dogs':
        kwargs = {'num_classes': 120}
        num_classes = 120
    elif args.dataset == 'fgvc':
        kwargs = {'num_classes': 100}
        num_classes = 100
    elif args.dataset == 'flowers':
        kwargs = {'num_classes': 102}
        num_classes = 102
    elif args.dataset == 'cars':
        kwargs = {'num_classes': 196}
        num_classes = 196
    elif args.dataset == 'cifar100':
        kwargs = {'num_classes': 100}
        num_classes = 100
    elif args.dataset == 'imagenet-s50':
        kwargs = {'num_classes': 50}
        num_classes = 50

    # create model
    if args.pretrained:
        if args.arch == 'resnet18':
            logger.info("=> using pre-trained model 'resnet18'")
            model = resnet.resnet18(pretrained=True)
        elif args.arch == 'resnet50':
            logger.info("=> using pre-trained model 'resnet50'")
            model = resnet.resnet50(pretrained=True)
        else:
            print('Arch not supported!!')
            exit()
    else:
        if args.arch == 'resnet18' :
            logger.info("=> creating model 'resnet18'")
            model = resnet.resnet18()
        elif args.arch == 'resnet50':
            logger.info("=> creating model 'resnet50'")
            model = resnet.resnet50()
        else:
            print('Arch not supported!!')
            exit()

    model = torch.nn.DataParallel(model)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.arch == 'resnet18':
        model.module.fc = nn.Linear(512, num_classes)
    elif args.arch == 'resnet50':
        model.module.fc = nn.Linear(2048, num_classes)
    model = model.cuda()

    logger.info(model)

    # define loss function (criterion) and optimizer
    xent_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    contrastive_criterion = NCESoftmaxLoss(args.t).cuda(args.gpu)

    cudnn.benchmark = True
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, val_dir_name)

    if args.dataset == 'cifar100':
        if not os.path.exists(os.path.join(args.data, 'train')):
            from download_datasets import download_and_prepare_cifar100
            download_and_prepare_cifar100(args.data)
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))

    elif args.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    elif args.dataset == 'imagenet-s50':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        raise NotImplementedError

    train_dataset = ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_batch_size = args.batch_size
    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    config_dict = vars(args)
    config_dict.update({
        "training_method": "CGC",
    })
    wandb.init(project="lico-reproduction", name="cgc", config=config_dict)
    
    if args.evaluate:
        validate(val_loader, model, contrastive_criterion, xent_criterion, args, logger)
        return

    best_save_path = None

    total_steps = len(train_loader) * args.epochs
    lr_scheduler = CosineLRScheduler(optimizer, T_max=total_steps)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # We are not adjusting every epoch, but every step
        # adjust_learning_rate(optimizer, epoch, args)
        
        # train for one epoch
        loss_epoch = train(train_loader, model, contrastive_criterion, xent_criterion, optimizer, epoch, args, logger, lr_scheduler)

        # evaluate on validation set
        acc1 = validate(val_loader, model, contrastive_criterion, xent_criterion, args, logger)
        
        wandb.log({
            "epoch": epoch,
            "train_loss_epoch": loss_epoch,
        }, step=global_step)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            best_save_path = save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.save_dir)
    
    # Upload best model to wandb
    logging.info(f"Uploading model at path {best_save_path} to wandb")
    wandb.save(best_save_path, policy='now')


def train(train_loader, model, contrastive_criterion, xent_criterion, optimizer, epoch, args, logger, lr_scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    xe_losses = AverageMeter('XE Loss', ':.4e')
    gc_losses = AverageMeter('GC Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, xe_losses, gc_losses, losses, top1, top5],
        logger,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    train_len = len(train_loader)
    train_iter = iter(train_loader)
    end = time.time()
    global global_step

    for i in range(train_len):
        # xe_images , images, aug_images, transforms_i, transforms_j, transforms_h, transforms_w, hor_flip, targets = train_iter.__next__()
        xe_images , images, aug_images, transforms_i, transforms_j, transforms_h, transforms_w, hor_flip, targets = train_iter.__next__()

        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        aug_images = aug_images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        aug_params_dict = {'transforms_i': transforms_i, 'transforms_j': transforms_j, 'transforms_h': transforms_h,
                            'transforms_w': transforms_w, 'hor_flip': hor_flip}

        aug_output, xe_loss, contrastive_loss = model(images, contrastive_criterion, xe_images=xe_images, aug_images=aug_images, 
                                                      aug_params_dict=aug_params_dict, targets=targets, xent_criterion=xent_criterion)
        xe_loss = xe_loss.mean()
        contrastive_loss = contrastive_loss.mean()

        loss = xe_loss + args.lambda_val * contrastive_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(aug_output, targets, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        xe_losses.update(xe_loss.item(), images.size(0))
        gc_losses.update(contrastive_loss.item(), images.size(0))

        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # CGC updates schedule per epoch, but LICO updates every step
        lr_scheduler.step()
        
        global_step += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    last_lr = lr_scheduler.get_last_lr()[0]

    wandb.log({
        "train_loss_step": xe_losses.avg,
        "train_cgc_part": gc_losses.avg,
        "trainer/global_step": global_step,
        "lr-SGD": last_lr,
    }, step=global_step)

    return losses.avg


def validate(val_loader, model, contrastive_criterion, criterion, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        logger,
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images,contrastive_criterion, vanilla=True)
            loss = criterion(output, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        wandb.log({
            "val_acc1": top1.avg / 100,
            "val_acc5": top5.avg / 100,
            "val_loss": losses.avg,
            # "val_cgc_part": contrastive_loss.item(),
        }, step=global_step)

    return top1.avg


def save_checkpoint(state, is_best, save_dir):
    epoch = state['epoch']
    filename = 'checkpoint_' + str(epoch).zfill(3) + '.pth.tar'
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)
    best_filename = 'model_best.pth.tar'
    best_save_path = os.path.join(save_dir, best_filename)
    if is_best:
        shutil.copyfile(save_path, best_save_path)
    if epoch % 25 == 0 and os.path.isfile(best_save_path):
        # Upload best model to wandb
        logging.info(f"Uploading model at path {best_save_path} to wandb")
        wandb.save(best_save_path, policy='now')
    return best_save_path


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters,logger, prefix="" ):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]

        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
