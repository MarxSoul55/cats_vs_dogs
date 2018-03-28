"""Visualizing the effect of average-pooling."""

import tensorflow as tf
import numpy as np
import cv2

image = cv2.imread('s:/mystuff/library/gallery/meme.jpg')
image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_NEAREST)
image = image.astype('float32')
data = np.array([image])
input_ = tf.placeholder(tf.float32, shape=[1, 1024, 1024, 3], name='input')
result = tf.layers.average_pooling2d(input_, 2, 2)
with tf.Session() as sess:
    result = sess.run(result, feed_dict={input_: data})
result = result.astype('uint8')
result = result[0]
print(result.shape)
cv2.imshow('cv2', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
