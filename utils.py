import tensorflow as tf
import numpy as np
import os
import glob
import random
import cv2

N_CLASSES = 2
BATCH_SIZE = 64
N_EPOCHS = 20
IMAGE_SIZE = 64


TRAIN_DATASET_PATH ='dataset/train/'
VAL_DATASET_PATH ='dataset/val/'


