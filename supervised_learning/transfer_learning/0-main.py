#!/usr/bin/env python3

""" import tensorflow.keras as K
preprocess_data = __import__('0-transfer').preprocess_data 
# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase 

model = K.applications.vgg16.VGG16()

model.summmary()
"""

""" (a, b), (X, Y) = K.datasets.cifar10.load_data()

print(a.shape)
print(b.shape) """
""" X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1) """
import tensorflow.keras as K
model = K.models.load_model("vgg19.h5")


model.trainable = True

set_trainable = False
for layer in model.layers:
    if layer.name in ['block5_conv1', 'block4_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

for layer in  model.layers:
    print(layer.name)
    print(layer.trainable)
model.summary()
model.save("vgg16-train.h5")
""" import tensorflow.keras as K
from keras.applications import vgg16
from keras.models import Model
import keras

vgg = K.applications.vgg16.VGG16()

output = vgg.layers[-4].output
vgg_model = Model(vgg.input, output) 

vgg_model.summary()
vgg_model.save("vgg19.h5") """

""" vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False
    
import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])   """