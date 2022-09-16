import os, sys
import joblib
dir1 = os.path.dirname(os.path.abspath(''))
dir2 = os.path.dirname(os.path.abspath('..'))
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)
from custom_scorer import prob_scorer
from sklearn import metrics
from ROC import ROCplot
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tikzplotlib

model_list = ['mvglm', 'svm_early', 'svm_late']
p_list = []
for m in model_list:
    model=m
    custom_transform = getattr(__import__('CVruns.{}'.format(model),
                                fromlist=[model]), 'custom_transform')
    custom_learner = getattr(__import__('CVruns.{}'.format(model),
                                fromlist=[model]), 'custom_learner')
    XX_test = getattr(__import__('CVruns.{}'.format(model),
                                fromlist=[model]), 'XX_test')
    y_test = getattr(__import__('CVruns.{}'.format(model),
                                fromlist=[model]), 'y_test')
    grid = joblib.load('../results/{}.pkl'.format(model))
    print(grid.best_score_)

    if 'text' in m:
        preds = grid.predict_proba(XX_test)[:,1]
    else:
        preds = grid.predict(XX_test)
    print('Test set AUC: {}'.format(metrics.roc_auc_score(y_test, preds)))
    p_list.append(preds)

pred_probs = p_list[0]
sns.distplot(np.array([pred_probs[i] for i,j in enumerate(y_test) if j ==0]),
             hist = False, kde = True, label='Negative',
             kde_kws = {'shade': True, 'linewidth': 3, 'cut': 0})
sns.distplot(np.array([pred_probs[i] for i,j in enumerate(y_test) if j ==1]),
             hist = False, kde = True, label='Positive'
             kde_kws = {'shade': True, 'linewidth': 3, 'cut': 0})
plt.legend(title=model_list[0], loc='upper left')
plt.show()
#tikzplotlib.save("densities2.tex")
