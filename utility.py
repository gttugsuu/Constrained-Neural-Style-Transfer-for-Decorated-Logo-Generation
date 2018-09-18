import numpy as np
import cv2


# Returns numpy array of input image (with channels last)
def load_image(path_to_img, height, width, invert):
    
    # Open image
    image = cv2.imread(path_to_img, 1)   
    
    # Invert if necessary
    if invert == 1:
        image = 255.0-image
    
    # Resize image
    image = cv2.resize(image, (height,width))
    print("image resized to ", image.shape) 
    
    # Add new axis
    image = image[np.newaxis,:,:,:]
    print("Image file shape is: ", image.shape)

    image = np.array(image, dtype = "float32")

    # Subtract optimization for VGG-Net
    image -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    # Return numpy array
    return image

# Saves JPEG image, inputs numpy array with shape = (1, height, width, depth)
def save_image(path,image, invert):
    # Output should add back the mean.
    image += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    # Get rid of the first useless dimension, what remains is the image.
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    print("Saved image file shape is: ", image.shape)
    # Convert from numpy to image
    if invert == 1:
        image = 255-image
    # Save image
    cv2.imwrite(path,image)