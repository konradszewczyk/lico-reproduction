import os

from PIL import Image, ImageDraw

# Class for images that should have a red dot added
red_dot_class = 'n02325366'  # wood rabbit, cottontail, cottontail rabbit

# Other classes not receiving a red dot
# All are living things, mostly encountered in greenish environments
other_classes = [
    'n01531178',  # goldfinch, Carduelis carduelis
    'n01644373',  # tree frog, tree-frog
    'n02104029',  # kuvasz
    'n02119022',  # red fox, Vulpes vulpes
    'n02123597',  # Siamese cat, Siamese
    'n02133161',  # American black bear, black bear, Ursus americanus, Euarctos americanus
    'n02165456',  # ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle
    'n02281406',  # sulphur butterfly, sulfur butterfly
    'n02483362',  # gibbon, Hylobates lar
]


def add_red_dot(image_path, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    dot_size = 4
    position = (0, 0)  # Top left corner
    draw.ellipse([position, (position[0] + dot_size, position[1] + dot_size)], fill='red')
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
    for cls in other_classes:
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
