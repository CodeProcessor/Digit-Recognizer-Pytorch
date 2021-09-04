#!/usr/bin/env python
"""
@Filename:    dataset.py
@Author:      dulanj
@Time:        2021-08-23 14.35
"""
import os

import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image


class DigitRecognizerDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transform=None, test=False):
        self.file_path = file_path
        self.dataframe = pd.read_csv(file_path)
        self.transform = transform
        self.is_test = test

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        _data = self.dataframe.iloc[index]
        _label = _data.label
        _numpy_array = np.array(_data.tolist()[1:])
        image = _numpy_array.reshape((28, 28)).astype('uint8')
        # cv2.imshow('disp', _numpy_2d)
        # cv2.waitKey(0)
        # print(_data)
        label_matrix = torch.zeros((10))
        label_matrix[_label] = 1
        if self.transform:
            image, label_matrix = self.transform(image, label_matrix)

        if self.is_test:
            return image, _label
        return image, label_matrix


if __name__ == '__main__':
    _filepath = "/home/dulanj/Datasets/Digit-Recognizer/digit-recognizer/train.csv"
    train_dataset = DigitRecognizerDataset(
        file_path=_filepath
    )
    print(len(train_dataset))
    train_dataset.main(10)

