import tensorflow as tf
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation

class DNNshape:
  def rect_dnn(self, inLayers=22, outLayers=2, numLayers=2, actFunc='relu', kerInit='normal', dropOut=0.2, outActFunc='softmax', multFactor=1):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(((inLayers-outLayers)*2), input_dim=(inLayers-outLayers)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(actFunc))
    model.add(tf.keras.layers.Dropout(dropOut))

    for i in range(0,numLayers):
      model.add(tf.keras.layers.Dense(((inLayers-outLayers)*2), kernel_initializer=kerInit))
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.Activation(actFunc))
      model.add(tf.keras.layers.Dropout(dropOut))

    model.add(tf.keras.layers.Dense(outLayers, kernel_initializer=kerInit))
    model.add(tf.keras.layers.Activation(actFunc))
    return model
  
  def diam_dnn(self, inLayers=22, outLayers=2, numLayers=2, actFunc='relu', kerInit='normal', dropOut=0.2, outActFunc='softmax', multFactor=1):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(((inLayers-outLayers)*2), input_dim=(inLayers-outLayers)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(actFunc))
    model.add(tf.keras.layers.Dropout(dropOut))

    for i in range(0,numLayers):
      model.add(tf.keras.layers.Dense(((inLayers-outLayers)/(i+1)), kernel_initializer=kerInit))
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.Activation(actFunc))
      model.add(tf.keras.layers.Dropout(dropOut))

    model.add(tf.keras.layers.Dense(outLayers, kernel_initializer=kerInit))
    model.add(tf.keras.layers.Activation(actFunc))
    return model
