import tensorflow as tf
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

'''
model.add(Dense(64, input_dim=14, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.5))
'''

class DNNshape:
  def rect_dnn(self, inLayers=22, outLayers=2, numLayers=2, actFunc='relu', kerInit='normal', dropOut=0.2, outActFunc='softmax', multFactor=1):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(((inLayers-outLayers)*2), input_dim=(inLayers-outLayers)))
    model.add(BatchNormalization())
    model.add(Activation(actFunc))
    model.add(tf.keras.layers.Dropout(dropOut))

    for i in range(0,numLayers):
      model.add(tf.keras.layers.Dense(((inLayers-outLayers)*2), kernel_initializer=kerInit))
      model.add(BatchNormalization())
      model.add(Activation(actFunc))
      model.add(tf.keras.layers.Dropout(dropOut))

    model.add(tf.keras.layers.Dense(outLayers, kernel_initializer=kerInit))
    model.add(Activation(actFunc))
    return model
  
  def diam_dnn(self, inLayers=22, outLayers=2, numLayers=2, actFunc='relu', kerInit='normal', dropOut=0.2, outActFunc='softmax', multFactor=1):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(((inLayers-outLayers)*2), input_dim=(inLayers-outLayers)))
    model.add(BatchNormalization())
    model.add(Activation(actFunc))
    model.add(tf.keras.layers.Dropout(dropOut))

    for i in range(0,numLayers):
      model.add(tf.keras.layers.Dense(((inLayers-outLayers)/(i+1)), kernel_initializer=kerInit))
      model.add(BatchNormalization())
      model.add(Activation(actFunc))
      model.add(tf.keras.layers.Dropout(dropOut))

    model.add(tf.keras.layers.Dense(outLayers, kernel_initializer=kerInit))
    model.add(Activation(actFunc))
    return model
