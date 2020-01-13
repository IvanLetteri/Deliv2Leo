from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

class ROCandCM:
  def rocCM(self, test_class, predict_class):
    fpr, recall, thresholds = roc_curve(test_class, predict_class)
    roc_auc = auc(false_positive_rate, recall)
    cm = confusion_matrix(test_class, predict_class)
    
    return fpr, recall, roc_auc, cm

