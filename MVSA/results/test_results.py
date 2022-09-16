import os, sys
import joblib
dir1 = os.path.dirname(os.path.abspath(''))
dir2 = os.path.dirname(os.path.abspath('..'))
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)
from custom_scorer import prob_scorer
from sklearn import metrics

model='mvglm'
custom_transform = getattr(__import__('CVruns.{}'.format(model),
                            fromlist=[model]), 'custom_transform')
custom_learner = getattr(__import__('CVruns.{}'.format(model),
                            fromlist=[model]), 'custom_learner')
XX_test = getattr(__import__('CVruns.{}'.format(model),
                            fromlist=[model]), 'XX_test')
y_test = getattr(__import__('CVruns.{}'.format(model),
                            fromlist=[model]), 'y_test')
grid = joblib.load('../results/{}.pkl'.format(model))
print('Test set size: {}'.format(y_test.shape[0]))
print(grid.best_score_)
print(grid.best_params_)

preds = grid.predict(XX_test)
print('Test set AUC: {}'.format(metrics.roc_auc_score(y_test, preds)))
