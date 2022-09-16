import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tikzplotlib
import os, sys
import joblib
dir1 = os.path.dirname(os.path.abspath(''))
dir2 = os.path.dirname(os.path.abspath('..'))
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)
from custom_scorer import median_score, MAP_score, lambda_score
import numpy as np
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
if 'mvglm' in model:
    XX_train = getattr(__import__('CVruns.{}'.format(model),
                            fromlist=[model]), 'XX')
    y_train = getattr(__import__('CVruns.{}'.format(model),
                            fromlist=[model]), 'y_train')
    Pipe = getattr(__import__('CVruns.{}'.format(model),
                            fromlist=[model]), 'Pipe')
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
df_cm = pd.DataFrame(metrics.confusion_matrix(y_test, preds.round(),
                     normalize='true')[::-1])
df_cm.index = list(range(12)[::-1])
sns.heatmap(df_cm.iloc[1:,:11], annot=False, cmap='Greens')
plt.yticks(rotation=0)
plt.xlabel('$y$')
plt.ylabel('$\\widehat{y}$')
plt.show()
#tikzplotlib.save('confusion.tex')
