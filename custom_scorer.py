from sklearn import metrics
import numpy as np
from collections import Counter
import math

def prob_scorer(y_true, probas):
    preds = np.round(probas)
    return metrics.accuracy_score(y_true, preds)

def late_scorer(y_true, pred_mat):
    late_preds = np.array([Counter(sorted(row, reverse=True)).most_common(1)[0][0]
                              if np.unique(row).shape[0] != 3
                              else np.random.choice(row)
                                      for row in pred_mat])
    return metrics.accuracy_score(y_true, late_preds)

def multinomial_accuracy(y_true, probas):
    arg_preds = np.argmax(probas, axis=1)
    return metrics.accuracy_score(y_true, arg_preds)

def lambda_score(y_true, preds):
    p = np.round(preds).squeeze()
    return metrics.mean_absolute_error(y_true, p)

def median_score(y_true, preds):
    p = np.floor(preds + 1/3 - 0.02*preds).squeeze()
    return metrics.mean_absolute_error(y_true, p)

def MAP_score(y_true, preds):
    pred_mat = np.zeros((preds.shape[0], 15))
    for i in range(preds.shape[0]):
        for j in range(15):
            pred_mat[i,j] = (preds[i]**j * np.exp(-preds[i]))/math.factorial(j)
    arg_preds = pred_mat.argmax(axis=1)
    return metrics.mean_absolute_error(y_true, arg_preds)
