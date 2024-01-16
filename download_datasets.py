import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import pickle
from PIL import Image


def download_and_prepare_cifar100(target_dir='./data'):
    # Download CIFAR100 dataset
    datasets.CIFAR100(target_dir, download=True, train=True)
    datasets.CIFAR100(target_dir, download=True, train=False)

    # Load CIFAR100 dataset
    with open(os.path.join(target_dir, 'cifar-100-python', 'train'), 'rb') as f:
        train_dict = pickle.load(f, encoding='bytes')
    with open(os.path.join(target_dir, 'cifar-100-python', 'test'), 'rb') as f:
        test_dict = pickle.load(f, encoding='bytes')

    # Create directories for train and test sets
    for i in range(100):
        os.makedirs(os.path.join(target_dir, 'train', str(i)), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'val', str(i)), exist_ok=True)

    # Save images in the appropriate directories
    for i in range(len(train_dict[b'data'])):
        img = Image.fromarray(train_dict[b'data'][i].reshape(3, 32, 32).transpose(1, 2, 0))
        img.save(os.path.join(target_dir, 'train', str(train_dict[b'fine_labels'][i]), str(i) + '.png'))

    for i in range(len(test_dict[b'data'])):
        img = Image.fromarray(test_dict[b'data'][i].reshape(3, 32, 32).transpose(1, 2, 0))
        img.save(os.path.join(target_dir, 'val', str(test_dict[b'fine_labels'][i]), str(i) + '.png'))
