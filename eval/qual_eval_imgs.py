
#         CIFAR-100
# Baseline x x x x
# LICO     x x x x

#         ImageNetS50
# Baseline x x x x
# LICO     x x x x

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob

# Function to get image paths from a folder
def get_image_paths(folder_path):
    return glob(folder_path + '/*.png')  # Update the extension if your images are in a different format

# Replace these folder paths with your actual folder paths
baseline_cifar100_folder = 'plot_imgs/4_imgs_baseline_cifar100'
lico_cifar100_folder = 'plot_imgs/4_imgs_lico_cifar100'

baseline_imagenets50_folder = 'plot_imgs/4_imgs_baseline_imagenets50'
lico_imagenets50_folder = 'plot_imgs/4_imgs_lico_imagenets50'

# Get image paths
baseline_cifar100_paths = get_image_paths(baseline_cifar100_folder)
lico_cifar100_paths = get_image_paths(lico_cifar100_folder)

baseline_imagenets50_paths = get_image_paths(baseline_imagenets50_folder)
lico_imagenets50_paths = get_image_paths(lico_imagenets50_folder)

# Organize image paths
cifar100_paths = baseline_cifar100_paths + lico_cifar100_paths
imagenets50_paths = baseline_imagenets50_paths + lico_imagenets50_paths

imgs_paths = {
    "CIFAR-100": cifar100_paths,
    "ImageNetS50": imagenets50_paths,
}

# Chosen classes
chosen_classes = {
    "CIFAR-100": ['bed', 'baby', 'boy', 'bottle'] * 2,
    "ImageNetS50": ['kuvasz', 'goldfinch', 'siamese cat', 'ladybug'] * 2,
}

fontsize = 26

for j, dataset in enumerate(['CIFAR-100', 'ImageNetS50']):
    # Plotting
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    
    classes = chosen_classes[dataset]
    
    fig.subplots_adjust(hspace=0.5)

    # CIFAR-100 images
    for i, path in enumerate(imgs_paths[dataset]):
        img = mpimg.imread(path)
        axs[i // 4, i % 4].imshow(img)
        if i // 4 == 0:
            axs[i // 4, i % 4].set_title(f'"{classes[i]}"', fontsize=fontsize)

    # fig.suptitle(f"Examples saliency maps for ResNet-18 on {dataset}", fontsize=fontsize)
    
    pad = 5

    axs[0, 0].set_ylabel('Baseline', rotation=90, fontsize=fontsize, labelpad=pad)
    axs[1, 0].set_ylabel('LICO', rotation=90, fontsize=fontsize, labelpad=pad)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.savefig(f"plot_imgs/quant_final_fig_{j}.png")
    # plt.show()
