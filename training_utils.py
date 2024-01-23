import logging
import torch

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


DATASETS_TO_CLASSES = {
    'cifar100': 100,
    'imagenet': 1000,
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
    'lamp', 'lawn mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
    'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
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

TEXT_CLASSES = {
    'cifar100': cifar100_classes,
    'imagenet-s50': imagenet_s50_classes,
    'imagenet': None,
    'cub': None,
    'aircraft': None,
    'flowers': None,
    'cars': None,
    'in9l': None
}
