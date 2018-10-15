import os
import math
from shutil import copyfile

RAW_IMAGES_FOLDER = "data/raw/"
TRAIN_IMAGES_FOLDER = "data/processed/train/"
VAL_IMAGES_FOLDER = "data/processed/val/"


def split_train_val_set(train_ratio=0.8):
    '''
    Stratified train/val split
    :param train_ratio:
    :return:
    '''
    categories = os.listdir(RAW_IMAGES_FOLDER)

    # TODO Check if already exist

    for cat in categories:
        cat_raw_path = RAW_IMAGES_FOLDER + cat + "/"
        files = os.listdir(cat_raw_path)
        n_train = math.floor(train_ratio * len(files))
        cat_train_dir = TRAIN_IMAGES_FOLDER + cat + "/"
        cat_val_dir = VAL_IMAGES_FOLDER + cat + "/"

        if not os.path.exists(cat_train_dir):
            os.makedirs(cat_train_dir)

        if not os.path.exists(cat_val_dir):
            os.makedirs(cat_val_dir)

        for i, file in enumerate(files, 1):
            if i > n_train:
                copyfile(cat_raw_path + file, cat_val_dir + file)
            else:
                copyfile(cat_raw_path + file, cat_train_dir + file)


if __name__ == '__main__':
    split_train_val_set()
