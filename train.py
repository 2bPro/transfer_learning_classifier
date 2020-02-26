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

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

# dimensions of our images (is this in px?)
img_width, img_height = 299, 299

train_data_dir = 'data/train' #location of training data
validation_data_dir = 'data/validate' #location of validation data

# number of samples used for determining the samples_per_epoch
nb_train_samples = 65
nb_validation_samples = 10
epochs = 20
batch_size = 5  

train_datagen = ImageDataGenerator(
        rescale=1./255,            # normalize pixel values to [0,1]
        shear_range=0.2,      
        zoom_range=0.2,    
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)       # normalize pixel values to [0,1]

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

model_top = Sequential()
model_top.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:], data_format=None)),  
model_top.add(Dense(256, activation='relu'))
model_top.add(Dropout(0.5))
model_top.add(Dense(1, activation='sigmoid')) 

model = Model(inputs=base_model.input, outputs=model_top(base_model.output))

model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size
)

model.reset_metrics()
model.save('classifier.h5')