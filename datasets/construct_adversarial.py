import os

from PIL import Image, ImageDraw

# Creates a variation of ImageNet-S with a red dot in the corner of all images of a certain class

# Class for images that should have a red dot added
red_dot_class = 'n02325366'  # wood rabbit, cottontail, cottontail rabbit


def add_red_dot(image_path, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    dot_size = (int(image.size[0] / 40), int(image.size[1] / 40))
    position = (0, 0)  # Top left corner
    draw.ellipse([position, (position[0] + dot_size[0], position[1] + dot_size[1])], fill='red')
    # should not be any compression
    image.save(output_path, quality=100)


def process_images(base_dir, adv_dir):
    sub_dirs = ['train', 'val']

    for sub_dir in sub_dirs:
        class_dir = os.path.join(base_dir, sub_dir, red_dot_class)
        if os.path.exists(class_dir):
            output_dir = os.path.join(adv_dir, sub_dir, f'red_dot_{red_dot_class}')
            os.makedirs(output_dir, exist_ok=True)

            for filename in os.listdir(class_dir):
                image_path = os.path.join(class_dir, filename)
                output_path = os.path.join(output_dir, filename)
                add_red_dot(image_path, output_path)


def create_symlinks(imagenet_s50_dir, adversarial_dir):
    for cls in os.listdir(os.path.join(imagenet_s50_dir, 'train')):
        if cls == red_dot_class:
            continue
        for sub_dir in ['train', 'val']:
            source_dir = os.path.join(imagenet_s50_dir, sub_dir, cls)
            symlink_dir = os.path.join(adversarial_dir, sub_dir, cls)

            # Check if the source directory exists
            if os.path.exists(source_dir):
                # Create parent directories for the symlink if they don't exist
                os.makedirs(os.path.dirname(symlink_dir), exist_ok=True)
                os.symlink(source_dir, symlink_dir)
            else:
                print(f"Source directory does not exist: {source_dir}")


if __name__ == '__main__':
    base_dir = 'C:/Users/Mikhail/Datasets/ImagenetS/ImageNetS50'
    adversarial_dir = 'C:/Users/Mikhail/Datasets/Adversarial'
    process_images(base_dir, adversarial_dir)
    # I didn't want to use additional space, so here are symlinks!
    create_symlinks(base_dir, adversarial_dir)
