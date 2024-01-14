import os
import pandas as pd
import shutil
from tqdm import tqdm


def move_images(csv_file, source_folder):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    for index, row in tqdm(df.iterrows()):
        class_name = row['Class']
        filename = row['Filename']
        source_file = os.path.join(source_folder, f'{filename}.JPEG')
        target_dir = os.path.join(source_folder, class_name)

        # Create target directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        target_file = os.path.join(target_dir, f'{filename}.JPEG')

        # Move the file
        if os.path.exists(source_file):
            shutil.move(source_file, target_file)
        else:
            print(f"File not found: {source_file}")


csv_file =      'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/imagenet_val_labels.csv'
source_folder = 'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val'

move_images(csv_file, source_folder)
