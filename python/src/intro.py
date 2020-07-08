import os
import tensorflow as tf
from PIL import Image
from tensorflow import keras

image_size = (180, 180)
testImagePath = "data/dog_test1.jpg"

img = keras.preprocessing.image.load_img(testImagePath, target_size=image_size)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 



print("************************************************************")
model = keras.models.load_model('./model/image_classification.h5')
print(model)
print("************************************************************")


predictions = model.predict(img_array)
score = predictions[0]
print(
    "This image is %.2f percent cat and %.2f percent dog."
    % (100 * (1 - score), 100 * score)
)

#docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./intro.py