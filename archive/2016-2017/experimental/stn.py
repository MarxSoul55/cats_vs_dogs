'''
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def load_img(data_path, desired_size=None, view=False):
    img = Image.open(data_path)
    img = img.convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    # preprocess
    x = np.asarray(img, dtype='float32')
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x

# ---------- PREPROCESSING ----------
# Parameters
DIMS = (400, 400)
CAT1 = 'cat1.jpg'
CAT2 = 'cat2.jpg'
# Load both images.
img1 = load_img(CAT1, DIMS)
img2 = load_img(CAT2, DIMS, view=True)
# Concatenate into (2, 400, 400, 3)
input_img = np.concatenate([img1, img2], axis=0)
# Get shape.
batch, height, width, channels = input_img.shape
# Initialize M to identity-transformation.
M = np.array([[1, 0, 0], [0, 1, 0]]).astype('float32')
# Reshaping M into (batch, 2, 3)
M = np.resize(m, (batch, 2, 3))
# ---------- STEP 1: THE MESHGRID ----------
x = np.linspace(-1, 1, width)
y = np.linspace(-1, 1, height)
x_t, y_t = np.mesgrid(x, y)
# Augment the dimensions to create homogeneous coordinates: reshape to (x_t, y_t, 1)
ones = np.ones(np.prod(x_t.shape))
sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
# Repeat grid `batch` times.
sampling_grid = np.resize(sampling_grid, (batch, 3, height * width))
'''
