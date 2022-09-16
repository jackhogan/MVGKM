import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import tikzplotlib

def ROCplot(Y, preds, model, tikz=False):
    plt.title('Receiver Operating Characteristic')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    print(zip(Y, preds, model))
    for y, p, m in zip(Y, preds, model):
        fpr, tpr, threshold = metrics.roc_curve(y, p)
        roc_auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr, label = '{}'.format(m))
    plt.legend(loc = 'lower right')
    if tikz:
        tikzplotlib.save("mytikz.tex")
    else:
        plt.show(block=True)
