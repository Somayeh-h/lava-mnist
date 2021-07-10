# A simple MNIST classifier using Lava

This module contains a simple 3 layer feedforward perceptron classifier for 
MNIST handwritten digits dataset.

The MLP consists of 784 LIF neurons at the input, **densely** connected to 
three subsequent LIF layers, consisting of 64, 64, and 10 neurons, 
respectively. The classification output is interpreted as the highest 
compartment voltage of a neuron in the last layer of 10 neurons.

The network was trained as an ANN in Keras and converted to an SNN using the 
SNN-toolbox (https://snntoolbox.readthedocs.io/en/latest/).

### Structure
- **datasets**: utility to download MNIST dataset using Tensorflow2/Keras 
  (https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data)
- **nets**: network architecture definition as a Lava Process
- **trained_models**: pre-trained saved weights; output of SNNToolBox
- **tutorials**: Jupyter tutorial(s)
