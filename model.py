import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.io

def load_vgg_model(path, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS):

    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']
    
    def weights(layer, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        """
        W = vgg_layers[0][layer][0][0][2][0][0]
        b = vgg_layers[0][layer][0][0][2][0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        
        assert layer_name == expected_layer_name, "expected %r, but given %r" % (expected_layer_name, layer_name)
        return W, b

    def relu(conv2d_layer):
        """
        Return the RELU function wrapped over a TensorFlow layer. Expects a
        Conv2d layer input.
        """
        return tf.nn.relu(conv2d_layer)

    def conv2d(prev_layer, layer, layer_name):
        """
        Return the Conv2D layer using the weights, biases from the VGG
        model at 'layer'.
        """
        W, b = weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def conv2d_relu(prev_layer, layer, layer_name):
        """
        Return the Conv2D + RELU layer using the weights, biases from the VGG
        model at 'layer'.
        """
        return relu(conv2d(prev_layer, layer, layer_name))

    def avgpool(prev_layer):
        """
        Return the AveragePooling layer.
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph = {}
    
    graph['input']    = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)), dtype = 'float32')
    
    graph['conv1_1']  = conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = avgpool(graph['conv1_2'])
    
    graph['conv2_1']  = conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = avgpool(graph['conv2_2'])
    
    graph['conv3_1']  = conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = avgpool(graph['conv3_4'])
    
    graph['conv4_1']  = conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = avgpool(graph['conv4_4'])
    
    graph['conv5_1']  = conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = avgpool(graph['conv5_4'])
    
    return graph