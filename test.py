from keras.models import load_model
from keras.preprocessing import image
import numpy as np

m = load_model('model.h5')

m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
test_image = image.load_img('testing_images/covid1 (4).jpeg', target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
r = m.predict(test_image)
print(r)

if (r[0][0] == 0):
    print ('Corona Positive')
else:
    print ('Pneumonia')    