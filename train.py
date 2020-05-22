from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator 

model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

dataset = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1./255)

training_dataset = dataset.flow_from_directory('dataset', 
                                               target_size = (224,224), 
                                               batch_size=32, class_mode = 
                                               'binary')

testing_dataset = dataset.flow_from_directory('testing_dataset', 
                                              target_size = (224,224), 
                                              batch_size=32, 
                                              class_mode = 'binary')

model.fit(training_dataset, 
          steps_per_epoch=250, 
          epochs=10, 
          validation_data=testing_dataset, 
          validation_steps=250)

model.save('model.h5')
