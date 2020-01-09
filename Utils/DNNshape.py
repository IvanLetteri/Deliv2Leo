

class DNNshape:
  def create_model(self, inLayers=22, outLayers=2, numLayers=2, actFunc='relu', kerInit='normal', dropOut=0.2, outActFunc='softmax', multFactor=1):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense((inLayers*2), input_dim=(inLayers-outLayers), activation=actFunc))
    model.add(tf.keras.layers.Dropout(dropOut))

    for i in range(0,numLayers):
      model.add(tf.keras.layers.Dense(inLayers, kernel_initializer=kerInit, activation=actFunc))
      model.add(tf.keras.layers.Dropout(dropOut))

    model.add(tf.keras.layers.Dense(outLayers, kernel_initializer=kerInit, activation=outActFunc))
    return model
