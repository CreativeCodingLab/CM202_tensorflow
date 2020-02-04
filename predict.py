import tensorflow as tf
import numpy as np

from model import *
from utils import *

test_image_path = 'dataset/test/Cat/8251.jpg'
test_image = cv2.resize(cv2.imread(test_image_path),(IMAGE_SIZE, IMAGE_SIZE))/255.
test_image_shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3)
test_image = tf.reshape(test_image, shape=test_image_shape)

model.load_weights('weights/')

prediction = model.predict(test_image)

print(prediction)

if prediction[0] < 0.5:
    print('Cat')
else:
    print('Dog')
