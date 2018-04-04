from keras.datasets import mnist
from keras.models import Sequential,load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import save_model
import h5py
from PIL import Image
import numpy as np


image=Image.open("inp1.png")
image=image.resize((28,28),Image.ANTIALIAS)
gray = image.convert('L')
bw = gray.point(lambda x: 0 if x<128 else 255, '1')
bw.save("result_bw.png")

image=Image.open("result_bw.png")
image=image.resize((28,28),Image.ANTIALIAS)
plt.imshow(image, cmap=plt.get_cmap('gray'))


data=list(image.getdata())
array=np.asarray(data)
array=array/255
arr=array.reshape(array.shape[0],-1)
arr=np.transpose(arr)

classifier=load_model('digits.h5')


classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

prediction=classifier.predict(arr)
predic=prediction[0]
max_val=-1
for i in range (0,10):
    if predic[i] > max_val:
        max_val=predic[i]
        predicted_val=i  
        
v=int(round(max_val*100)) 
print('\n\nI think the result is '+str(predicted_val)+' with '+str(v)+' probability.')
