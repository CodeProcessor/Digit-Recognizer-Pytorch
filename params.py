#!/usr/bin/env python
"""
@Filename:    params.py.py
@Author:      dulanj
@Time:        2021-08-23 14.22
"""
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHANNELS=3
NO_OF_CLASSES = 10
MODEL_SAVE_PATH='trained_models/best-model-parameters.pt'

LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
WEIGHT_DECAY = 0
EPOCHS = 250
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = 'trained_models/best-model-parameters.pt'
TRAIN_CSV = '/home/dulanj/Datasets/Digit-Recognizer/digit-recognizer/train.csv'
IMG_TEST_DIR = '/home/dulanj/Datasets/dogs-vs-cats/test1'
