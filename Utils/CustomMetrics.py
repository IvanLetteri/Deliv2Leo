from keras import backend as K
class CustomMetrics:
  def __init__(self, customCallback = ['accuracy']):
    self.customCallback = ['accuracy']
    
  def setCallback(self, mattCorr, precision, fmeasure, recall):
    if mattCorr:
      self.customCallback.append('mattCorr')
    if precision:
      self.customCallback.append('precision')
    if fmeasure:
      self.customCallback.append('fmeasure')
    if recall:
      self.customCallback.append('recall')
    
    return self.customCallback  

  #matthews_correlation
  def mcor(self, y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
  
  def precision(self, y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
  
  def recall(self, y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
  
  def f1(self, y_true, y_pred):
    def recall(self, y_true, y_pred):
      true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
      possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
      recall = true_positives / (possible_positives + K.epsilon())
      return recall
    
    def precision(self, y_true, y_pred):
      true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
      predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
      precision = true_positives / (predicted_positives + K.epsilon())
      return precision
    
    precision = precision(self, y_true, y_pred)
    recall = recall(self, y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
  
