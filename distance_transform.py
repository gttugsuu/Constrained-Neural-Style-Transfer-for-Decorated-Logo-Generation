# %load distance_transform.py
"""
Inputs images with black background as as Tensor
with shape of (1, height, width, 3)
"""
import cv2
import numpy as np

"""
Inputs (1, height, width, 3) tensor.
Outputs (height, width) as grayscale image
"""

def numpy_to_image(image):

	image = image*1.0
	# Remove VGG optimization
	# if image.shape == (1,400,400,3):
	if image.ndim == 4:
		image += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
		# Cut unneeded axis
		image = image[0]
	elif image.ndim == 3:
	# elif image.shape == (400,400,3):
		image += np.array([123.68, 116.779, 103.939]).reshape((1,1,3))
	
	image = np.clip(image, 0,255).astype("uint8")

	# Make grayscale
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	return image_gray

"""
Inputs 3-channel image, 
returns 1. Distance transformation with background as (+), foreground as (-)
		2. Sum of all elements in pixel-wise multiply of dist_t and original image

"""
def dist_t(image):
	
	# Grayscale image
	image_gray = numpy_to_image(image)

	# Binary image
	# The threshold results will be a tuple, with [value, image]
	image_th = cv2.threshold(image_gray,50,255,cv2.THRESH_BINARY)[1]
	# cv2.imwrite("image_f_dist.jpg", image_th)

	# Invert images
	image_inv = (255-image_th)

	### Distance transform

	# Distance transformation of background
	dist_t_bg = cv2.distanceTransform(image_inv, cv2.DIST_L2, 3)

	# Distance transformation of characters or patterns
	dist_t_fg = cv2.distanceTransform(image_th, cv2.DIST_L2, 3)

	# Make new distance transformation with (-) inside & (+) background
	dist_template = dist_t_bg + dist_t_fg*(-1)
    
    # Image float
	image_float = image_gray/255.0

	# Multiply pixel-wise with dist1
	mult = np.multiply(image_float, dist_template)

	# Sum of all elements in multiplied mult
	dist_sum = np.sum(mult)

	# return only dist_t_g as dist_template, dist_sum
	return dist_t_bg, dist_sum
	
"""
Assign 1s to input image characters,
and calculate distance loss as pixel wise
"""

def dist_loss(image, dist_template, orig_sum):

	# Grayscale image
	image_gray = numpy_to_image(image)
	cv2.imwrite("image_gray.jpg", image_gray)

	image_float = image_gray/255.0

	# Multiply pixel-wise with dist1
	mult = np.multiply(image_float, dist_template)

	# Sum of all elements in multiplied mult
	dist_sum = np.sum(mult)

	# Absolute value of difference between orig_sum & dist_sum
	dist_loss = abs(orig_sum - dist_sum)
	
	# with open("output.txt", "w") as f:
	# 	f.write(dist_sum)
	# 	f.write(dist_loss)

	return dist_loss