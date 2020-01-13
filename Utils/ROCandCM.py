class ROCandCM:
  def mcor(self, test_class, predict_class):
    false_positive_rate, recall, thresholds = roc_curve(test_class, predict_class)
    roc_auc = auc(false_positive_rate, recall)
    cm = confusion_matrix(test_class, predict_class)
    
    return roc_auc, cm

