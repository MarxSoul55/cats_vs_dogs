from PIL import Image, ImageCms
import numpy as np

image = Image.open('s:/mystuff/projects/cv/cats_vs_dogs/src/data/test_small/1.jpg')
if image.mode != 'RGB':
    print('Uh... Not RGB wtf?')
    image = image.convert('RGB')
srgb_profile = ImageCms.createProfile("sRGB")
lab_profile = ImageCms.createProfile("LAB")
rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB",
                                                            "LAB")
lab_im = ImageCms.applyTransform(image, rgb2lab_transform)
tensor = np.array(lab_im)
print(tensor.shape)
