import keras.losses

class CustomLossFunction:
  # Huber loss function        
  def huber_loss(y_true, y_pred, clip_value=1):
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
    return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
      import tensorflow as tf
      if hasattr(tf, 'select'):
        return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
      else:
        return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))

#keras.losses.huber_loss = huber_loss

