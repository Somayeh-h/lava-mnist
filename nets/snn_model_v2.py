import os
import sys
import numpy as np
sys.path.insert(1, str('./lava-core'))
from lava.core.generic.process import Process
from lava.processes.generic.dense import Dense
from lava.processes.generic.lif import LIF
from lava.core.generic.enums import Backend



class DatasetBuffer:
    def __init__(self, **kwargs):
        dataset_path = kwargs.pop('dataset_path', 'datasets/Nordland')
        self.out_size = kwargs.pop('out_size', 784)

        x_train = np.load(dataset_path + '/train_frames.npy')
        x_test = np.load(dataset_path + '/test_frames.npy')
        y_train = np.load(dataset_path + '/inputNumbers50.npy')

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = np.array(list(y_train)*2)
        self.y_test = y_train



class vprExecVarIO(Process):
    """A simple place categorisation classifying process, comprised by only
    LIF and Dense processes. The architecture is: Input (784,) -> Dense (400,)
    """

    def get_matrix_from_file(self, fileName):
        '''
        Loads a weight matrix from given file name, shaped based on number of input neurons, 
        and number of connections 
        '''

        if "XeAe" in fileName:
            n_tgt = n_e
            n_src = n_input  
        else:
            n_tgt = n_e
            n_src = n_i

        readout = np.load(fileName)
        print(readout.shape, fileName)
        value_arr = np.zeros((n_src, n_tgt))

        if not readout.shape == (0,):
            value_arr[np.int32(readout[:, 0]), np.int32(readout[:, 1])] = readout[:,2]
        return value_arr


    def _build(self, **kwargs):

        dataset_path = kwargs.pop('dataset_path', './datasets/Nordland')
        trained_wgts = kwargs.pop('trained_weights_path', './XeAe.npy')

        w_list = np.load(trained_wgts, allow_pickle=True)

        w_dense = self.get_matrix_from_file(trained_wgts)
        w_dense = w_dense.transpose() 

        # scale values as in mnist example 
        w_dense = np.interp(w_dense, (w_dense.min(), w_dense.max()), (-255, 255)).astype(np.int32)
        print(w_dense)

        b_lif = np.zeros((400,), dtype=np.int32)
        self.data_buff = DatasetBuffer(dataset_path=dataset_path, out_size=784)

        self.input_lif = LIF(size=784, bias_exp=6, vth=1528, dv=0, du=4095)
        self.dense = Dense(wgts=w_dense)
        self.lif = LIF(size=self.dense.size[0], bias_mant=b_lif, bias_exp=6, vth=380, dv=0, du=4095)

        self.dense(s_in=self.input_lif.s_out)
        self.lif(a_in=self.dense.a_out)


    def reset_process(self):
            """This is a helper method for classify() method. It zeros-out the
            neural states using ExecVars, instead of using SequentialProcesses"""
            self.input_lif.bias_mant.value = np.ones((self.input_lif.size,), dtype=np.int32)

            self.input_lif.u.value = np.zeros((self.input_lif.size,),
                                            dtype=np.int32)
            self.input_lif.v.value = np.zeros((self.input_lif.size,),
                                            dtype=np.int32)
            self.lif.u.value = np.zeros((self.lif.size,), dtype=np.int32)


    @staticmethod
    def compute_new_biases(raw_x_data, img_id, size):
        """This is a helper method for classify() method"""

        new_img_bias = np.int32((raw_x_data[img_id, :]) * 255)
        new_img_bias = new_img_bias.reshape((size,))

        return new_img_bias



    def classify(self, backend=Backend.TF, n_img=50, n_st_img=512):
        """Directly uses ExecVars (not SequentialProcesses) to inject 
        input and read output. Robust and slow - needs to stop Nx
        hardware after every image to read output and send input 
        """

        predictions = []
        ground_truths = []

        self.run(num_steps=0, backend=backend, eagerly=True)

        for img_id in range(n_img):

            new_biases = self.compute_new_biases(raw_x_data=self.data_buff.x_test, img_id=img_id, size=self.data_buff.out_size)

            self.input_lif.bias_mant.value = new_biases

            self.run(num_steps=n_st_img, backend=backend, eagerly=True)

            voltages = self.lif.v.value
            predictions.append(np.argmax(voltages))
            ground_truths.append(np.argmax(self.data_buff.y_test[img_id]))

            self.reset_process()

            if backend == Backend.N2:
                self._executor.board.sync = True
                self._executor.board.push()
            
            print("Processed: {}".format(img_id))

        if backend == Backend.N2:
            self.disconnect()

        accuracy = np.sum(np.array(predictions) == np.array( ground_truths)) / len(ground_truths) * 100
        print(predictions, ground_truths)
        print(f'{backend.name} execution was {accuracy} % accurate.')






if __name__ == '__main__':
    data_path = './lava-mnist/datasets/Nordland'
    wgts_path = './lava-mnist/trained_models/XeAe.npy'
    num_steps_per_image = 512
    num_images = 50

    n_e = 400
    n_i = 400
    n_input = 784  

    os.environ['SLURM'] = '1'

    print("SLURM:", os.environ['SLURM'])

    vpr_classifier = vprExecVarIO(dataset_path=data_path, trained_weights_path=wgts_path)

    vpr_classifier.classify(backend=Backend.TF, n_img=num_images, n_st_img=num_steps_per_image)

