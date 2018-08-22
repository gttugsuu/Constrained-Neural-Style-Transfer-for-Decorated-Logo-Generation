# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 20:59:50 2017

@author: GT
"""
import tensorflow as tf
import numpy as np
import cv2
import os
import time
import argparse

import distance_transform
import utility
import model

###############################################################################
# Constants for the image input and output.
###############################################################################

# Output folder for the images.
OUTPUT_DIR = 'output/'

# Content image to use.
content_input_path = "input/contents/"
content_with_ext   = "humans1.bmp"
content_image_path = content_input_path + content_with_ext
content_image      = content_with_ext[:-4]

# Style image to use.
style_input_path   = "input/styles/"
style_with_ext     = "colorful_flower.jpg"
style_image_path   = style_input_path + style_with_ext
style_image        = style_with_ext[:-4]

# Image dimensions constants. 
# image = Image.open(content_image_path)  
#IMAGE_WIDTH = image.size[0]
IMAGE_WIDTH = 400
IMAGE_HEIGHT = IMAGE_WIDTH
COLOR_CHANNELS = 3

# Invertion of images
content_invert = 1
style_invert = 1
result_invert = content_invert
###############################################################################
# Algorithm constants
###############################################################################

# Number of iterations to run.
ITERATIONS = 5000
# path to weights of VGG-19 model
VGG_MODEL = "../imagenet-vgg-verydeep-19.mat"
# The mean to subtract from the input to the VGG model. 
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

parser = argparse.ArgumentParser(description='A Neural Algorithm of Artistic Style')
parser.add_argument('--w1', '-w1', default='1',help='w1')
parser.add_argument('--w2', '-w2', default='1',help='w2')
parser.add_argument('--w3', '-w3', default='1',help='w3')
parser.add_argument('--w4', '-w4', default='1',help='w4')
parser.add_argument('--w5', '-w5', default='1',help='w5')
parser.add_argument("--IMAGE_WIDTH", "-width", default = 400, help = "width & height of image")
parser.add_argument("--CONTENT_IMAGE", "-CONTENT_IMAGE", default = content_image_path, help = "Path to content image")
parser.add_argument("--STYLE_IMAGE", "-STYLE_IMAGE", default = style_image_path, help = "Path to style image")

parser.add_argument("--alpha",  "-alpha",   default="0.001",   help="alpha")
parser.add_argument("--beta",   "-beta",    default="0.8",     help="beta")
parser.add_argument("--gamma",  "-gamma",   default="0.001",    help="gamma")
args = parser.parse_args()

# Style image layer weights
w1 = float(args.w1)
w2 = float(args.w2)
w3 = float(args.w3)
w4 = float(args.w4)
w5 = float(args.w5)

# Content & Style weights
alpha = float(args.alpha)
beta = float(args.beta)
gamma = float(args.gamma)

CONTENT_IMAGE = str(args.CONTENT_IMAGE)
STYLE_IMAGE = str(args.STYLE_IMAGE)

# Splitting content path & name
dot = 0
slash = 0
for c in reversed(CONTENT_IMAGE):
    dot += 1
    if c == ".":
        break
for c in reversed(CONTENT_IMAGE):
    slash += 1 
    if c =="/" or c =="\\":
        break
content_path = CONTENT_IMAGE[:1-slash]
content_name = CONTENT_IMAGE[1-slash:-dot]

# Splitting style path & name
dot = 0 
slash = 0 
for c in reversed(STYLE_IMAGE):
    dot += 1
    if c == ".":
        break
for c in reversed(STYLE_IMAGE):
    slash += 1
    if c == "/" or c =="\\":
        break
style_path = STYLE_IMAGE[:1-slash]
style_name = STYLE_IMAGE[1-slash:-dot]

###############################################################################

def style_loss_func(sess, model):
    """
    Style loss function as defined in the paper.
    """
    def gram_matrix(F, N, M):
        """
        The gram matrix G.
        """
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def style_loss(a, x):
        """
        The style loss calculation.
        """
        # N is the number of filters (at layer l).
        N = a.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = a.shape[1] * a.shape[2]
        # A is the style representation of the original image (at layer l).
        A = gram_matrix(a, N, M)
        # G is the style representation of the generated image (at layer l).
        G = gram_matrix(x, N, M)
        result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result

    # Layers to use. We will use these layers as advised in the paper.
    # To have softer features, increase the weight of the higher layers
    # (conv5_1) and decrease the weight of the lower layers (conv1_1).
    # To have harder features, decrease the weight of the higher layers
    # (conv5_1) and increase the weight of the lower layers (conv1_1).
    layers = [
            ('conv1_2', w1),
            ('conv2_2', w2),
            ('conv3_2', w3),
            ('conv4_2', w4),
            ('conv5_2', w5),
            ]

    E = [style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in layers]
    W = [w for _, w in layers]
    loss = sum([W[l] * E[l] for l in range(len(layers))])
    return loss

def content_loss_func(sess, model):
    """
    Content loss function as defined in the paper.
    """
    def content_loss(p, x):
        return 0.5 * tf.reduce_sum(tf.pow(x - p, 2))
    loss = content_loss(sess.run(model['conv4_2']), model['conv4_2'])
    return loss

def content_dist(sess, model):

    dist, dist_sum = distance_transform.dist_t(sess.run(model["input"]))
    return tf.convert_to_tensor(dist, dtype=tf.float32), dist_sum

def shape_loss_func(sess, model, dist_template, dist_sum):

    content_image = sess.run(model['input'])
    mixed_image   = model["input"]

    # Convert to grayscale
    content_image = tf.image.rgb_to_grayscale(content_image)
    mixed_image   = tf.image.rgb_to_grayscale(mixed_image)

    # Remove dimensions of size 1 from the shape of a tensor
    content_image = tf.squeeze(content_image)
    mixed_image   = tf.squeeze(mixed_image)

    # Pixel-wise multiplication
    content_dist  = content_image * dist_template
    mixed_dist    = mixed_image   * dist_template

    print(content_dist.shape, mixed_dist.shape, dist_template.shape)

    loss_tensor = 0.5 * tf.reduce_sum(tf.pow(content_dist-mixed_dist, 2))

    return loss_tensor
    
if __name__ == '__main__':
    try:
        OUTPUT_DIR = ("output/" + content_name + "_vs_" + style_name)
        os.mkdir(OUTPUT_DIR)
    except:
        pass
    start_time = time.time()
    
    with tf.device("/gpu:0"):
        with tf.Session() as sess:          

            # Load images.
            content_image = utility.load_image(CONTENT_IMAGE, OUTPUT_DIR+'/'+content_with_ext, IMAGE_HEIGHT, IMAGE_WIDTH, invert = content_invert)
            style_image   = utility.load_image(STYLE_IMAGE, OUTPUT_DIR+'/'+style_with_ext, IMAGE_HEIGHT, IMAGE_WIDTH, invert = style_invert)
            utility.save_image(OUTPUT_DIR+"/"+style_name+".png", style_image, invert = style_invert)
            
            # Load the model.
            model = model.load_vgg_model(VGG_MODEL, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
            # Content image as input image
            initial_image = content_image
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            
            # Construct content_loss using content_image.
            sess.run(model['input'].assign(content_image))
            content_loss = content_loss_func(sess, model)

            # Construct shape loss using content image
            sess.run(model["input"].assign(initial_image))
            dist_template_inf, content_dist_sum = distance_transform.dist_t(content_image)
            ### take power of distance template
            dist_template = np.power(dist_template_inf,8)
            dist_template[dist_template>np.power(2,30)] = np.power(2,30)
            print(dist_template.sum())
            shape_loss = shape_loss_func(sess, model, dist_template, content_dist_sum)
    
            # Construct style_loss using style_image.
            sess.run(model['input'].assign(style_image))
            style_loss = style_loss_func(sess, model)
            
            # Instantiate equation 7 of the paper.
            total_loss = alpha * content_loss + beta * style_loss + gamma * shape_loss
    
            # Then we minimize the total_loss, which is the equation 7.
            optimizer = tf.train.AdamOptimizer(1.0)
            train_step = optimizer.minimize(total_loss)
    
            sess.run(tf.global_variables_initializer())
            sess.run(model['input'].assign(initial_image))
            for it in range(ITERATIONS+1):
                sess.run(train_step)
                
                if it%100 == 0:
                    # Print every 100 iteration.
                    mixed_image = sess.run(model['input'])
                    print('Iteration %d' % (it))
                    print('sum         : ', sess.run(tf.reduce_sum(mixed_image)))
                    print('total_loss  : ', sess.run(total_loss))
                    print("content_loss: ", alpha*sess.run(content_loss))
                    print("style_loss  : ", beta *sess.run(style_loss))
                    print("shape loss  : ", gamma*sess.run(shape_loss))
    
                    if not os.path.exists(OUTPUT_DIR):
                        os.mkdir(OUTPUT_DIR)
    
                    filename = OUTPUT_DIR + '/%d.jpg' % (it)
                    utility.save_image(filename, mixed_image, invert = result_invert)
                if sess.run(total_loss) < 1:
                    break
        sess.close()
    end_time = time.time()
    print("Time taken = ", end_time - start_time)