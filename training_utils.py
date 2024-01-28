import logging
import os

import torch
import torchvision.transforms as transforms

from datasets.imagenet_classes import imagenet_classes


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False, log_current_file=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    if log_current_file:
        logger.info(filepath)
        with open(filepath, "r") as f:
            logger.info(f.read())

        for f in package_files:
            logger.info(f)
            with open(f, "r") as package_f:
                logger.info(package_f.read())

    return logger


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def get_normalize_and_testdir(args):
    if args.dataset == 'cifar100':
        if not os.path.exists(os.path.join(args.data, 'train')):
            from download_datasets import download_and_prepare_cifar100
            download_and_prepare_cifar100(args.data)
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        if args.batch_size != 64:
            print('batch size is not 64 on CIFAR!')

        # cifar100 has only 2 sets of data
        testdir = os.path.join(args.data, 'val')

    elif args.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if args.batch_size != 128:
            print('batch size is not 128 on ImageNet!')
        testdir = os.path.join(args.data, 'val')

    elif args.dataset == 'imagenet-s50':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if args.batch_size != 128:
            print('batch size is not 128 on ImageNet-S!')
        testdir = os.path.join(args.data, 'val')

    elif args.dataset == 'adversarial':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        testdir = os.path.join(args.data, 'val')

    else:
        raise NotImplementedError
    return normalize, testdir


DATASETS_TO_CLASSES = {
    'cifar100': 100,
    'imagenet': 1000,
    'imagenet-s50': 50,
    'adversarial': 10,
    'cub': 200,
    'aircraft': 100,
    'flowers': 102,
    'cars': 196,
    'in9l': 9
}

cifar100_classes = (
    'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy',
    'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur', 'dolphin',
    'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain',
    'mouse',
    'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal',
    'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
    'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
)

imagenet_s50_classes = (
    "goldfish", "tiger shark", "goldfinch", "tree frog", "kuvasz", "red fox", "siamese cat", "american black bear",
    "ladybug", "sulphur butterfly", "wood rabbit", "hamster", "wild boar", "gibbon", "african elephant", "giant panda",
    "airliner", "ashcan", "ballpoint", "beach wagon", "boathouse", "bullet train", "cellular telephone", "chest",
    "clog", "container ship", "digital watch", "dining table", "golf ball", "grand piano", "iron", "lab coat",
    "mixing bowl", "motor scooter", "padlock", "park bench", "purse", "streetcar", "table lamp", "television",
    "toilet seat", "umbrella", "vase", "water bottle", "water tower", "yawl", "street sign", "lemon", "carbonara",
    "agaric"
)

adversarial_classes = (
    "goldfinch", "tree frog", "kuvasz", "red fox",
    "siamese cat", "american black bear",
    "ladybug", "sulphur butterfly", "gibbon",
    # rabbit goes last because it is the red dot class,
    # it's folder is named differently
    # (torch dataset sorts alphabetically)
    "wood rabbit",
)

TEXT_CLASSES = {
    'cifar100': cifar100_classes,
    'imagenet-s50': imagenet_s50_classes,
    'imagenet': imagenet_classes,
    'adversarial': adversarial_classes,
    'cub': None,
    'aircraft': None,
    'flowers': None,
    'cars': None,
    'in9l': None
}
