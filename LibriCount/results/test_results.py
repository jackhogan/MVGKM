import os, sys
import joblib
dir1 = os.path.dirname(os.path.abspath(''))
dir2 = os.path.dirname(os.path.abspath('..'))
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)
from custom_scorer import median_score, MAP_score, lambda_score
import numpy as np
from sklearn import metrics

model='mvml'
custom_transform = getattr(__import__('CVruns.{}'.format(model),
                           fromlist=[model]), 'custom_transform')
custom_learner = getattr(__import__('CVruns.{}'.format(model),
                           fromlist=[model]), 'custom_learner')
XX_test = getattr(__import__('CVruns.{}'.format(model), fromlist=[model]),
                  'XX_test')
y_test = getattr(__import__('CVruns.{}'.format(model), fromlist=[model]),
                 'y_test')
grid = joblib.load('../results/{}.pkl'.format(model))
print('Test set size: {}'.format(y_test.shape[0]))
print('CV best score: {}'.format(grid.best_score_))
print('CV best params: {}'.format(grid.best_params_))

if 'mvglm' in model:
    XX_train = getattr(__import__('CVruns.{}'.format(model),
                        fromlist=[model]), 'XX')
    y_train = getattr(__import__('CVruns.{}'.format(model),
                        fromlist=[model]), 'y_train')
    Pipe = getattr(__import__('CVruns.{}'.format(model), fromlist=[model]),
                   'Pipe')
    best_idx = np.argwhere(
                 grid.cv_results_['rank_test_{}'.format('lambda')] == 1
                 )[0][0]
    best_params = grid.cv_results_['params'][best_idx]
    Pipe.set_params(
            clf__lmbda = best_params['clf__lmbda'],
            clf__gamma = best_params['clf__gamma'],
            clf__wPen = best_params['clf__wPen']
            )
    est = Pipe.fit(XX_train, y_train)
    preds = est.predict(XX_test)
    print('Lambda Test set MAE: {}'.format(lambda_score(y_test, preds)))
else:
    preds = grid.predict(XX_test)
    print('Test set MAE: {}'.format(lambda_score(y_test, preds)))
