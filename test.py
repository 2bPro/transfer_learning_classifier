'''Making use of TensorFlow and Keras in order to train a model to distinguish between
   abdominal or chest radiology images. This training is done based on transfer learning
   from the Inception V3 model.

   The training as well as the testing are based on the paper entitled "Hello World Deep
   Learning in Medical Imaging" by Paras Lakhani et al.

   Directory structure:
       data -> train -> abdomen -> training images of the abdomen
                     -> chest -> training images of the chest
            -> validate -> abdomen -> validation images of the abdomen
                        -> chest -> validation images of the chest
            -> test -> combination of images of the abdomen and/or chest

   Log:
       2020-02-24 - BP - Initial creation based on paper
                       - Split training and testing into two separate scripts
                       - Persist training by saving model and reusing it for testing
'''
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras import models


# dimensions of our images (is this in px?)
img_width, img_height = 299, 299

abdomen_test = 'data/test/chest6.png'
chest_test = 'data/test/chest5.png'

model = models.load_model('classifier.h5')

# process image 1
image1 = image.load_img(chest_test, target_size=(img_width, img_height))

plt.imshow(image1)
plt.show()

image1 = image.img_to_array(image1)
x = np.expand_dims(image1, axis=0) * 1./255
score = model.predict(x)

print('Predicted:', score, 'Abdominal X-ray' if score < 0.5 else 'Chest X-ray')

# process image 2
image2 = image.load_img(abdomen_test, target_size=(img_width, img_height))

plt.imshow(image2)
plt.show()

image2 = image.img_to_array(image2)
x = np.expand_dims(image2, axis=0) * 1./255
score2 = model.predict(x)

print('Predicted:', score2, 'Abdominal X-ray' if score2 < 0.5 else 'Chest X-ray')