from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.engine.topology import Input
from keras import optimizers

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np

from keras.utils import plot_model
# plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

# model = VGG16(include_top=True, weights='imagenet', 
#                 input_tensor=None, input_shape=None, pooling=None, classes=1000)
model_vgg16_conv = VGG16(include_top=False, weights='imagenet')
print(model_vgg16_conv.layers)
# model_vgg16_conv.trainable = False
for layer in model_vgg16_conv.layers[:-4]:
    layer.trainable = False
model_vgg16_conv.summary()

# creating my own input format
input = Input(shape=(224,224,3),name = 'image_input')

# Use the above generated model
output_vgg16_conv = model_vgg16_conv(input)

# Adding fully-connected layers
x = Flatten(name='flatten1')(output_vgg16_conv)
print("flattened x.shape = ",x.shape)
x = Dense(4096, activation='relu', name='fc1')(x)
print("fc1 x.shape = ",x.shape)
# x = Dense(2048, activation='relu', name='fc2')(x)
# print("fc2 x.shape = ",x.shape)
x = Dense(1024, activation='relu', name='fc3')(x)
print("fc3 x.shape = ",x.shape)
x = Dense(64, activation='relu', name='fc4')(x)
print("fc3 x.shape = ",x.shape)
x = Dense(2, activation='softmax', name='predictions')(x)
print("predictions x.shape = ",x.shape)

# Creating my new model
my_model = Model(input=input, output=x)
my_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
my_model.summary()

plot_model(my_model, to_file="model.png", show_shapes=True, show_layer_names=True)
