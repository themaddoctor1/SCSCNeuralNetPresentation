# Neural Network Presentation
#### Given to the Stevens Computer Science Club on 3/29/2017

This repository contains the materials used in my presentation on Neural Networks that I gave to
the Stevens Computer Science club.

### How to use code

The perceptron demo provided has 3 modes: and, xor and random, which
operate roughly as expected. Random takes two arguments in the described
order: the number of points and the separation factor. A separation factor of 1 creates
linearly separable data, while 0 generally creates non-separable data.

The backpropagation demo includes several flags that can be used to customize the
experience. The -h flag can be used to view these arguments. The -s flag allows
for the network to be customized. In the code, I required the first layer to take
3 inputs and the last layer to output one value. The format is to give the number of layers,
a set of layer sizes (the size of which is one greater in size than the number of layers), and
a set of transfer functions. For instance, the default arguments of the program are equivalent
to running

`python code/backprop.py -d and -s 2 3 3 1 sigmoid linear`

This allows a user to play around with the different values. The user is able to set any network size
that takes 3 inputs and gives one output, and has the linear, sigmoid, relu and unit step functions
available for experimentation. Feel free to play around with the code and the command line to explore
these abilities and features.

NOTE: I am not certain as of writing this of whether or not the alpha modifier work.
