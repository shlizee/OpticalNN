"""
replace first layer of AlexNet with optical setup
adopt other layers of AlexNet code from AlexNet.py, code written by Frederik Kratzert at
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
"""
import tensorflow as tf
import numpy as np
Iterator = tf.data.Iterator


# xx, yy, Lambda, k_z_values can all be generated from onn_setup.ipynb.
xx = np.load('xx.npy')
yy = np.load('yy.npy')
Lambda = np.load('Lambda.npy')
k_z_values = np.load('k_z_values.npy')
x_tensor = tf.constant(xx, tf.float32)
y_tensor = tf.constant(yy, tf.float32)
Lambda_tensor = tf.constant(Lambda, tf.float32)
k_z = tf.constant(k_z_values, tf.complex64)
f = tf.constant(0.3E-2)


def fftshift_tf(data):
    """
    :param data: input tensor to do fftshift
    :return: after fftshift
    """
    dims = tf.shape(data)
    num = dims[3]
    shift_amt = (num - 1) / 2
    shift_amt = tf.cast(shift_amt, np.int32)
    output = tf.manip.roll(data, shift=shift_amt, axis=2)
    output = tf.manip.roll(output, shift=shift_amt, axis=3)

    return output


def ifftshift_tf(data):
    """
    Performs an ifftshift operation on the last two dimensions of a 4-D input tensor
    :param data: input tensor to do ifftshift
    :return: after ifftshift
    """
    dims = tf.shape(data)
    num = dims[3]
    shift_amt = (num + 1) / 2
    shift_amt = tf.cast(shift_amt, np.int32)
    output = tf.manip.roll(data, shift=shift_amt, axis=2)
    output = tf.manip.roll(output, shift=shift_amt, axis=3)

    return output


def generate_phase():
    """
    Generates the phase for a lens based on the focal length variable "f".
    Other referenced variables are global
    :return: phase generated
    """
    phase = tf.constant(2 * np.pi, tf.float32)\
            / Lambda * (tf.sqrt(tf.square(x_tensor) + tf.square(y_tensor) + tf.square(f)) - f)
    phase = tf.cast(phase, tf.complex64)
    return phase


def generate_propagator():
    """
    Generates the Fourier space propagator based on the focal length variable "f".
    Other referenced variables are global
    :return: propagator generated
    """
    propagator = tf.exp(1j * k_z * tf.cast(f, tf.complex64))
    propagator = ifftshift_tf(propagator)

    return propagator


def propagate(input_field, propagator):
    """
    Propagate an input E-field distribution along the optical axis using the defined propagator
    :param input_field: input field for doing propagation
    :param propagator: generated propagator
    :return: result after propagation
    """
    output = tf.ifft2d(tf.fft2d(input_field) * propagator)

    return output


def simulate_4f_system(input_field, kernel):
    """
    Pass an image through a 4f system
    :param input_field: input field of our 4f system
    :param kernel: kernel for doing convolution
    :return: output of our 4f system
    """
    # Calculate the lens phase
    lens_phase = generate_phase()

    # Calculate the propagator
    propagator = generate_propagator()

    # Propagate up to the first lens
    before_l1 = propagate(input_field, propagator)

    # Apply lens1 and propagate to the filter plane
    before_kernel = propagate(before_l1 * tf.keras.backend.exp(-1j * lens_phase), propagator)

    # Apply kernel and propagate to the second lens
    before_l2 = propagate(before_kernel * kernel, propagator)

    # Apply lens2 and propagate to the output plane
    output = propagate(before_l2 * tf.keras.backend.exp(-1j * lens_phase), propagator)

    # Return output of the 4f optical convolution
    return output


def convolve_with_all_kernels(image, batch_size, name):
    """
    doing convolution with all kernels in frequency domain
    :param image: input image
    :param kernel_in: kernel for doing convolution
    :param name: scope name
    :return: result after doing convolution with all kernels
    """
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name='weights', trainable=True,
                                 shape=[11, 11, 3, 96])
    #         f = tf.get_variable(name ='f', initializer=0.3E-2, trainable=True)
    # Zero pad the kernels for subsequent Fourier processing
    kernels = tf.concat([kernel, tf.constant(np.zeros((11, 216, 3, 96)), tf.float32)], axis=1)
    kernels = tf.concat([kernels, tf.constant(np.zeros((216, 227, 3, 96)), tf.float32)], axis=0)

    # Align the kernels for Fourier transforming
    kernels = tf.transpose(kernels, perm=[3, 2, 0, 1])
    kernels = tf.cast(kernels, tf.complex64)
    kernels = tf.fft2d(kernels)
    kernels = ifftshift_tf(kernels)

    # Add an extra dimension for the batch size and duplicate
    # the kernels to apply equally to all images in the batch
    kernels = tf.expand_dims(kernels, axis=0)
    kernels = tf.tile(kernels, multiples=[batch_size, 1, 1, 1, 1])

    # Add a dimension to the input image tensor to
    # enable convolution with all 96 first layer kernels
    image = tf.cast(image, tf.complex64)
    image = tf.expand_dims(image, axis=1)
    image = tf.transpose(image, perm=[0, 1, 4, 2, 3])
    image = tf.tile(image, multiples=[1, 96, 1, 1, 1])

    # Simulate the 4f system output for all 96 kernels
    # for all color channels and sum the channel outputs
    output = tf.reduce_sum(tf.abs(simulate_4f_system(image, kernels)) ** 2, axis=2)

    # Transpose and flip the output for display purposes
    output = tf.transpose(output, perm=[0, 2, 3, 1])
    output = tf.image.flip_left_right(output)
    output = tf.image.flip_up_down(output)

    # Convert to float format
    output = tf.cast(output, tf.float32)

    # Return the output
    return output


class ONN(object):
    """Implementation of ONN based on AlexNet implementation."""

    def __init__(self, x, batch_size, keep_prob, num_classes, skip_layer, weights_path='DEFAULT'):
        """Create the graph of the ONN model.

        Args:
            x: Placeholder for the input tensor.
            batch_size: batch_size for training
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.BATCH_SIZE = batch_size
        self.SKIP_LAYER = skip_layer
        self.KEEP_PROB = keep_prob
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'kernel_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""
        # 1st Layer: OP-conv
        conv1 = convolve_with_all_kernels(self.X, self.BATCH_SIZE, 'op')
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 27 * 27 * 256])
        fc6 = fc(flattened, 27 * 27 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

    def load_initial_weights(self, session):
        """
        load pre-trained kernel from first layer of AlexNet
        """
        # Load the weights into memory
        kernel_pretrained = np.load(self.WEIGHTS_PATH, encoding='bytes')
        with tf.variable_scope('op', reuse=True):
            # Assign kernel to its corresponding tf variable
            var = tf.get_variable('weights', trainable=False)
            session.run(var.assign(kernel_pretrained))


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', trainable=True, shape=[filter_height,
                                                                    filter_width,
                                                                    input_channels / groups,
                                                                    num_filters])
        biases = tf.get_variable('biases', trainable=True, shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', shape=[num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)
