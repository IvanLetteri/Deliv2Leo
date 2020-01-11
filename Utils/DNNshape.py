import tensorflow as tf
from keras.models import Sequential

class DNNshape:
  def rect_dnn(self, inLayers=22, outLayers=2, numLayers=2, actFunc='relu', kerInit='normal', dropOut=0.2, outActFunc='softmax', multFactor=1):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense((inLayers*2), input_dim=(inLayers-outLayers), BatchNormalization(), activation=actFunc))
    model.add(tf.keras.layers.Dropout(dropOut))

    for i in range(0,numLayers):
      model.add(tf.keras.layers.Dense((inLayers*2), kernel_initializer=kerInit, BatchNormalization(), activation=actFunc))
      model.add(tf.keras.layers.Dropout(dropOut))

    model.add(tf.keras.layers.Dense(outLayers, kernel_initializer=kerInit, BatchNormalization(), activation=outActFunc))
    return model
  
  def diam_dnn(self, inLayers=22, outLayers=2, numLayers=2, actFunc='relu', kerInit='normal', dropOut=0.2, outActFunc='softmax', multFactor=1):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense((inLayers*2), input_dim=(inLayers-outLayers), BatchNormalization(), activation=actFunc))
    model.add(tf.keras.layers.Dropout(dropOut))

    for i in range(0,numLayers):
      model.add(tf.keras.layers.Dense((inLayers/(i+1)), kernel_initializer=kerInit, BatchNormalization(), activation=actFunc))
      model.add(tf.keras.layers.Dropout(dropOut))

    model.add(tf.keras.layers.Dense(outLayers, kernel_initializer=kerInit, BatchNormalization(), activation=outActFunc))
    return model
