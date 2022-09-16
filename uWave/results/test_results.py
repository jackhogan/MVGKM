import os, sys
import joblib
dir1 = os.path.dirname(os.path.abspath(''))
dir2 = os.path.dirname(os.path.abspath('..'))
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)
from custom_scorer import multinomial_accuracy, late_scorer
from sklearn import metrics

model='mvml'
custom_transform = getattr(__import__('CVruns.{}'.format(model),
                            fromlist=[model]), 'custom_transform')
custom_learner = getattr(__import__('CVruns.{}'.format(model),
                            fromlist=[model]), 'custom_learner')
XX_test = getattr(__import__('CVruns.{}'.format(model),
                            fromlist=[model]), 'XX_test')
y_test = getattr(__import__('CVruns.{}'.format(model),
                            fromlist=[model]), 'y_test')
grid = joblib.load('{}.pkl'.format(model))
print('Test set size: {}'.format(y_test.shape[0]))
print(grid.best_score_)
print(grid.best_params_)

preds = grid.predict(XX_test)
if 'mvglm' in model:
    print('Test set accuracy: {}'.format(multinomial_accuracy(y_test, preds)))
elif 'late' in model:
    print('Test set accuracy: {}'.format(late_scorer(y_test, preds)))
else:
    print('Test set accuracy: {}'.format(metrics.accuracy_score(y_test,preds)))
