import os, sys
import joblib
dir1 = os.path.dirname(os.path.abspath(''))
dir2 = os.path.dirname(os.path.abspath('..'))
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)
from custom_scorer import prob_scorer
from sklearn import metrics
from ROC import ROCplot

model_list = ['mvglm', 'svm_early', 'svm_late']
y_list = []
p_list = []
for m in model_list:
    model=m
    custom_transform = getattr(__import__('CVruns.{}'.format(model), fromlist=[model]), 'custom_transform')
    custom_learner = getattr(__import__('CVruns.{}'.format(model), fromlist=[model]), 'custom_learner')
    XX_test = getattr(__import__('CVruns.{}'.format(model), fromlist=[model]), 'XX_test')
    y_test = getattr(__import__('CVruns.{}'.format(model), fromlist=[model]), 'y_test')
    grid = joblib.load('../results/{}_AUC.pkl'.format(model))
    print('Test set size: {}'.format(y_test.shape[0]))
    print('Proportion 1s: {}'.format(y_test.sum()/y_test.shape[0]))
    print(grid.best_score_)
    print(grid.best_params_)
    y_list.append(y_test)

    if 'text' in m:
        preds = grid.predict_proba(XX_test)[:,1]
    else:
        preds = grid.predict(XX_test)
    if 'mvglm' in model:
        print('Test set accuracy: {}'.format(prob_scorer(y_test, preds)))
    else:
        print('Test set accuracy: {}'.format(metrics.accuracy_score(y_test, preds.round())))
    print('Test set AUC: {}'.format(metrics.roc_auc_score(y_test, preds)))
    p_list.append(preds)

ROCplot(Y=y_list, preds=p_list, model=model_list, tikz=True)
