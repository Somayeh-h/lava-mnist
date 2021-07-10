#Copyright (C) 2021 Intel Corporation
#SPDX-License-Identifier:  BSD-3-Clause

import os

import numpy as np
from lava.core.generic.process import Process
from lava.processes.generic.dense import Dense
from lava.processes.generic.lif import LIF
from lava.core.generic.enums import Backend


class DatasetBuffer:
    def __init__(self, **kwargs):
        dataset_path = kwargs.pop('dataset_path', 'datasets/MNIST')
        self.out_size = kwargs.pop('out_size', 784)

        x_train_npz = np.load(dataset_path + '/x_train.npz')
        x_test_npz = np.load(dataset_path + '/x_test.npz')
        y_train_npz = np.load(dataset_path + '/y_train.npz')
        y_test_npz = np.load(dataset_path + '/y_test.npz')

        self.x_train = x_train_npz['arr_0']
        self.x_test = x_test_npz['arr_0']
        self.y_train = y_train_npz['arr_0']
        self.y_test = y_test_npz['arr_0']


class MnistExecVarIO(Process):
    """A simple MNIST digit classifying process, comprised by only LIF and
    Dense processes. The architecture is: Input (784,) -> Dense (64,
    ) -> Dense(64,) -> Dense(10,)"""
    def _build(self, **kwargs):
        dataset_path = kwargs.pop('dataset_path', './datasets/MNIST')
        trained_wgts = kwargs.pop('trained_weights_path',
                                  './mnist_dense_only_snn_wblist.npy')
        wb_list = np.load(trained_wgts, allow_pickle=True)
        w_dense1 = wb_list[0].transpose().astype(np.int32)
        b_lif1 = wb_list[1].astype(np.int32)
        w_dense2 = wb_list[2].transpose().astype(np.int32)
        b_lif2 = wb_list[3].astype(np.int32)
        w_dense3 = wb_list[4].transpose().astype(np.int32)
        b_lif3 = wb_list[5].astype(np.int32)

        self.data_buff = DatasetBuffer(dataset_path=dataset_path,
                                       out_size=784)
        self.input_lif = LIF(size=784, bias_exp=6, vth=1528,
                             dv=0, du=4095)
        self.dense1 = Dense(wgts=w_dense1)
        self.lif1 = LIF(size=self.dense1.size[0], bias_mant=b_lif1,
                        bias_exp=6, vth=380, dv=0, du=4095)
        self.dense2 = Dense(wgts=w_dense2)
        self.lif2 = LIF(size=self.dense2.size[0], bias_mant=b_lif2,
                        bias_exp=6, vth=354, dv=0, du=4095)
        self.dense3 = Dense(wgts=w_dense3)
        self.lif3 = LIF(size=self.dense3.size[0], bias_mant=b_lif3,
                        bias_exp=6, vth=131071, dv=0, du=4095)

        self.dense1(s_in=self.input_lif.s_out)
        self.lif1(a_in=self.dense1.a_out)
        self.dense2(s_in=self.lif1.s_out)
        self.lif2(a_in=self.dense2.a_out)
        self.dense3(s_in=self.lif2.s_out)
        self.lif3(a_in=self.dense3.a_out)

    def reset_process(self):
        """This is a helper method for classify() method. It zeros-out the
        neural states using ExecVars, instead of using SequentialProcesses"""
        self.input_lif.bias_mant.value = np.ones((self.input_lif.size,),
                                                  dtype=np.int32)
        self.input_lif.u.value = np.zeros((self.input_lif.size,),
                                          dtype=np.int32)
        self.input_lif.v.value = np.zeros((self.input_lif.size,),
                                          dtype=np.int32)
        self.lif1.u.value = np.zeros((self.lif1.size,), dtype=np.int32)
        self.lif1.v.value = np.zeros((self.lif1.size,), dtype=np.int32)
        self.lif2.u.value = np.zeros((self.lif2.size,), dtype=np.int32)
        self.lif2.v.value = np.zeros((self.lif2.size,), dtype=np.int32)
        self.lif3.u.value = np.zeros((self.lif3.size,), dtype=np.int32)
        self.lif3.v.value = np.zeros((self.lif3.size,), dtype=np.int32)

    @staticmethod
    def compute_new_biases(raw_x_data, img_id, size):
        """This is a helper method for classify() method"""
        new_img_bias = np.int32((raw_x_data[img_id, :]) * 255)
        new_img_bias = new_img_bias.reshape((size,))
        return new_img_bias

    def classify(self, backend=Backend.TF, n_img=10, n_st_img=512):
        """This function uses ExecVars directly, instead of
        SequentialProcesses to inject input and read output. It is robust,
        but slow, for it has to stop Nx hardware from executing after every
        image to read the output and inject the input."""
        predictions = []
        ground_truths = []

        self.run(num_steps=0, backend=backend)
        for img_id in range(n_img):
            new_biases = self.compute_new_biases(
                raw_x_data=self.data_buff.x_test, img_id=img_id,
                size=self.data_buff.out_size)
            self.input_lif.bias_mant.value = new_biases
            self.run(num_steps=n_st_img, backend=backend)
            voltages = self.lif3.v.value
            predictions.append(np.argmax(voltages))
            ground_truths.append(np.argmax(self.data_buff.y_test[img_id]))
            self.reset_process()
            if backend == Backend.N2:
                self._executor.board.sync = True
                self._executor.board.push()

        if backend == Backend.N2:
            self.disconnect()
        print(predictions, ground_truths)
        accuracy = np.sum(np.array(predictions) == np.array(
            ground_truths)) / len(ground_truths) * 100
        print(f'{backend.name} execution was {accuracy} % accurate.')


if __name__ == '__main__':
    data_path = '../datasets/MNIST'
    wgts_path = '../trained_models/mnist_dense_only_snn_wblist.npy'
    num_steps_per_image = 512
    num_images = 10

    os.environ['SLURM'] = '1'

    mnist_classifier = MnistExecVarIO(dataset_path=data_path,
                                      trained_weights_path=wgts_path,
                                      num_steps_per_image=num_steps_per_image,
                                      total_num_images=num_images)

    mnist_classifier.classify(backend=Backend.TF, n_img=num_images,
                              n_st_img=num_steps_per_image)

