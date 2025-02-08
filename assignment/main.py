import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import shutil
import random
import tensorflow as tf

cat_path = "C:/Users/erand/OneDrive - University of Jaffna/myGit/Cat-and-Dog-detection-ML-model/assignment/dataset/Cats"
dog_path = "C:/Users/erand/OneDrive - University of Jaffna/myGit/Cat-and-Dog-detection-ML-model/assignment/dataset/Dogs"
ds = "C:/Users/erand/OneDrive - University of Jaffna/myGit/Cat-and-Dog-detection-ML-model/assignment/dataset"

for folder in ['train', 'validation', 'test']:
    for subfolder in ['cats', 'dogs']:
        os.makedirs(os.path.join(ds, folder, subfolder), exist_ok=True)

def split_data(source, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.2, 0.1)):
    files = os.listdir(source)
    random.shuffle(files)
    train_size = int(len(files) * split_ratio[0])
    val_size = int(len(files) * split_ratio[1])
    test_size = len(files) - train_size - val_size

    for i, file in enumerate(files):
        if i < train_size:
            shutil.copy(os.path.join(source, file), os.path.join(train_dir, file))
        elif i < train_size + val_size:
            shutil.copy(os.path.join(source, file), os.path.join(val_dir, file))
        else:
            shutil.copy(os.path.join(source, file), os.path.join(test_dir, file))

split_data(cat_path, 
           os.path.join(ds, 'train', 'cats'), 
           os.path.join(ds, 'validation', 'cats'), 
           os.path.join(ds, 'test', 'cats'))

split_data(dog_path, 
           os.path.join(ds, 'train', 'dogs'), 
           os.path.join(ds, 'validation', 'dogs'), 
           os.path.join(ds, 'test', 'dogs'))

print(len(os.listdir(cat_path)))
print(len(os.listdir(dog_path)))
print(len(os.listdir(os.path.join(ds, 'train', 'cats'))))
print(len(os.listdir(os.path.join(ds, 'train', 'dogs'))))
print(len(os.listdir(os.path.join(ds, 'validation', 'cats'))))
print(len(os.listdir(os.path.join(ds, 'validation', 'dogs'))))
print(len(os.listdir(os.path.join(ds, 'test', 'cats'))))
print(len(os.listdir(os.path.join(ds, 'test', 'dogs'))))

