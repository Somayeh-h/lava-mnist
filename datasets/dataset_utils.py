#Copyright (C) 2021 Intel Corporation
#SPDX-License-Identifier:  BSD-3-Clause

import os

import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.datasets.mnist as mnist
import matplotlib.pyplot as plt


class DatasetUtil:
    """This class is a convenience utility to fetch MNIST data using Keras
    API, save the data in compressed NumPy files (.npz extension), and save
    MNIST digits as PNG files after reading them one by one from NumPy
    compressed archive (.npz extension)."""
    def __init__(self, **kwargs):
        self.dataset_path = kwargs.pop('dataset_path', '../datasets/MNIST')
        self.dataset_type = kwargs.pop('dataset_type', 'train')
        self.first_image_idx = kwargs.pop('first_image_idx', 0)
        self.total_num_images = kwargs.pop('total_num_images', 25)
        self.normalize_x = kwargs.pop('normalize_input', True)
        self.flatten_x = kwargs.pop('flatten_input', True)
        self.one_hot_y = kwargs.pop('one_hot_labels', True)

        (self.x_train, self.y_train), (self.x_test, self.y_test) = \
            mnist.load_data()
        if self.flatten_x:
            self.x_train = self.x_train.reshape((60000, 784))
            self.x_test = self.x_test.reshape((10000, 784))

        if self.normalize_x:
            self.x_train = (self.x_train / 255) - 0.5
            self.x_test = (self.x_test / 255) - 0.5

        if self.one_hot_y:
            self.y_train = keras.utils.to_categorical(self.y_train,
                                                      num_classes=10)
            self.y_test = keras.utils.to_categorical(self.y_test,
                                                     num_classes=10)

    def save_npz(self):
        """Saves training and test data in separate NumPy compressed files (
        .npz extension)"""
        if not os.path.exists(self.dataset_path):
            print(f'Creating {os.path.realpath(self.dataset_path)}')
            os.makedirs(self.dataset_path, exist_ok=True)
        else:
            print(f'Dataset directory {os.path.realpath(self.dataset_path)} '
                  f'already exists')

        np.savez_compressed(os.path.join(self.dataset_path, 'x_train.npz'),
                            arr_0=self.x_train)
        np.savez_compressed(os.path.join(self.dataset_path, 'y_train.npz'),
                            arr_0=self.y_train)
        np.savez_compressed(os.path.join(self.dataset_path, 'x_test.npz'),
                            arr_0=self.x_test)
        np.savez_compressed(os.path.join(self.dataset_path, 'y_test.npz'),
                            arr_0=self.y_test)

    def save_digit_images(self):
        if self.dataset_type == 'train':
            x = self.x_train
        elif self.dataset_type == 'test':
            x = self.x_test
        else:
            raise ValueError('Invalid dataset type provided. Dataset type '
                             'should be test or train')
        img_dir = os.path.realpath(self.dataset_path + '/../' +
                                   self.dataset_type + '_images')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        for j in range(self.first_image_idx, self.first_image_idx +
                       self.total_num_images):
            img = x[j, :]
            img = img.reshape(x.shape[1], 1)
            plt.imsave(img_dir + '/' + str(j) + '.png', img)

    def save_labels_txt(self):
        if self.dataset_type == 'train':
            y = self.y_train
        elif self.dataset_type == 'test':
            y = self.y_test
        else:
            raise ValueError('Invalid dataset type provided. Dataset type '
                             'should be test or train')
        label_file_name = self.dataset_type + '_labels.txt'
        labels = y[self.first_image_idx:self.first_image_idx +
                   self.total_num_images, :]
        np.savetxt(label_file_name, labels, fmt='%d')


if __name__ == '__main__':
    db = DatasetUtil(dataset_path='./MNIST',
                     dataset_type='test',
                     first_image_idx=0,
                     total_num_images=100)
    db.save_npz()
    db.save_digit_images()
    db.save_labels_txt()
