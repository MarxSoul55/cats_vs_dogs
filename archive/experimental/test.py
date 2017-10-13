import numpy as np
import PIL
from PIL import Image

image = Image.open('D:/MyStuff/Programming/machine_learning/cv/cats_dogs/data/train/cats/cat.7.jpg')
image = image.resize([256, 256], resample=PIL.Image.LANCZOS)
image = image.convert('RGB')
# image.show()
gimage = image.convert('L')
# gimage.show()
image = np.array(image).astype('float32')
gimage = np.array(gimage).astype('float32')
gimage = gimage[..., np.newaxis]
print(image.shape)
print(gimage.shape)
c = np.concatenate([image, gimage], axis=2)
print(c.shape)
# print(image)
# print(np.max(image, axis=1))


'''
old preprocess

------------------------------

def preprocess(image):
    """
    Preprocesses an image for the model.
    Converts image to 256x256x4 (RGB & Grayscale).
    It is meant to resemble rods/cones in the retina.

    # Parameters
        image (str): Path to the image.
    # Returns
        A preprocessed image (numpy array).
    """
    image = Image.open(image)
    image = image.resize([DIM, DIM], resample=PIL.Image.LANCZOS)
    image = image.convert('HSV')
    image = np.array(image).astype('float32')
    image /= 255
    return image
'''
