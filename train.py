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
          steps_per_epoch=65, 
          epochs=3, 
          validation_data=testing_dataset, 
          validation_steps=65)

scores = model.evaluate(testing_dataset, verbose=1)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

test_loss, test_acc = model.evaluate(testing_dataset)
print('Test loss', test_loss)
print('Test accuracy', test_acc)

with open('Output.txt', 'a') as f:
    print("accuracy=>",test_acc, file=f)

if(test_acc > 0.85):
    exit 0
else:
    exit 1


