# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from keras.layers import Input, Convolution2D, merge
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.optimizers import Adam
#from keras import optimizers
import matplotlib.pyplot as plt
import h5py
import math
import keras

#for save excel:
import xlwt
from tempfile import TemporaryFile
    
def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    
    """
    return 10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)

def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.1
	epochs_drop = 20
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

	##paramaters

batch_size = 64
nb_epoch = 60

#input imaage dimensions
img_rows, img_cols = 31, 31
out_rows, out_cols = 31, 31


#.....................................................

#load data:

##load train data
file = h5py.File('data/train_291_aug_scale3_blur1.h5', 'r')
in_train = file['data'][:]
out_train = file['label'][:]
file.close()
# ...............

#load validation data
file = h5py.File('data/test_scale3_blur1.h5', 'r')
in_test = file['data'][:]
out_test = file['label'][:]
file.close()

# .........................................................

#convert data form 
in_train = in_train.astype('float32')
out_train = out_train.astype('float32')
in_test = in_test.astype('float32')
out_test = out_test.astype('float32')
#print('folatx=', tf.floatx())


#if K.image_dim_ordering() == 'tf':
if K.image_data_format() == 'channels_last':      
    in_train = in_train.reshape(in_train.shape[0], img_rows, img_cols, 1)
    in_test  = in_test.reshape(in_test.shape[0], img_rows, img_cols, 1)
    out_train = out_train.reshape(out_train.shape[0], out_rows, out_cols, 1)
    out_test = out_test.reshape(out_test.shape[0], out_rows, out_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    

#print number of training patches
print( 'k.image=',   K.image_data_format())
print('in_train shape:', in_train.shape)
print(in_train.shape[0], 'train samples')
print(in_test.shape[0], 'test samples')
print('input_shape=', input_shape)

#SR Model
#input tensor for a 1_channel image region
x = Input(shape = input_shape)
c1 = Convolution2D(64, (9, 9), padding="same", kernel_initializer="he_normal", activation="relu")(x)
c2 = Convolution2D(32, (5, 5), padding="same", kernel_initializer="he_normal", activation="relu")(c1)
m= keras.layers.concatenate([c1, c2])
c3 = Convolution2D(32, (5, 5), padding="same", kernel_initializer="he_normal", activation="relu")(m)
c4 = Convolution2D(32, (5, 5), padding="same", kernel_initializer="he_normal", activation="relu")(c3)
c5 = Convolution2D(32, (5, 5), padding="same", kernel_initializer="he_normal", activation="relu")(c4)
c6 = Convolution2D(32, (5, 5), padding="same", kernel_initializer="he_normal", activation="relu")(c5)
c7 = Convolution2D(32, (5, 5), padding="same", kernel_initializer="he_normal", activation="relu")(c6)
c8 = Convolution2D(1, (5, 5), padding="same", kernel_initializer="he_normal")(c7)
model = Model(inputs = x, outputs = c8)

print("model=", model.summary())
##compile
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
model.compile(loss='mean_squared_error', metrics=[PSNRLoss], optimizer=adam) #'mse'


# learning schedule callback
lrate = LearningRateScheduler(step_decay)
#print('lrate=',lrate)
callbacks_list = [lrate]
history = model.fit(in_train, out_train, batch_size=batch_size, epochs=nb_epoch, callbacks = [lrate],
          verbose=1, validation_data=(in_test, out_test))            
print(history.history.keys())


#save model and weights
json_string = model.to_json()  
open('dbsr_model.json','w').write(json_string)  
model.save_weights('dbsr_model_weights_blur1.h5')
model.save_weights('dbsr_model_blur1.h5')

# summarize history for Peak signal to noise ratio (PSNR)
plt.figure()
plt.plot(history.history['PSNRLoss'])
plt.plot(history.history['val_PSNRLoss'])
plt.title('Peak Signal to Noise Ratio')
plt.ylabel('PSNR (dB)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
#plt.show()
plt.grid(b=True, which='major', ls='-')
plt.grid(b=True, which='minor', ls='-', alpha=0.2)
plt.minorticks_on()
plt.savefig('Epoch and PSNR SR.png')

# for save excel file:
book = xlwt.Workbook()
sheet1 = book.add_sheet('sheet1')
for i,e in enumerate(history.history['val_PSNRLoss']):
    sheet1.write(i,1,e)

name = "PSNR_Validation.xls"
book.save(name)
book.save(TemporaryFile())

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
#plt.show()
plt.grid(True,which="both",ls="-")
plt.savefig('Epoch and Loss SR.png')
