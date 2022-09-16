import os, sys
import joblib
dir1 = os.path.dirname(os.path.abspath(''))
dir2 = os.path.dirname(os.path.abspath('..'))
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)
from custom_scorer import median_score, MAP_score, lambda_score
import numpy as np
from sklearn import metrics
from mvglm_algo import MVGLM
import matplotlib.pyplot as plt
import tikzplotlib


tikz=True
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

XX_train = getattr(__import__('CVruns.{}'.format(model),
                            fromlist=[model]), 'XX')
y_train = getattr(__import__('CVruns.{}'.format(model),
                            fromlist=[model]), 'y_train')
best_idx = np.argwhere(grid.cv_results_['rank_test_score'] == 1)[0][0]
best_params = grid.cv_results_['params'][best_idx]
gmma_list = [1/(2*np.mean(metrics.pairwise_distances(XX_train.get_view(i)))**2)
             for i in range(3)]
est = MVGLM(lmbda = best_params['clf__lmbda'],
            gamma = best_params['clf__gamma'],
            wPen = best_params['clf__wPen'],
            learn_w=True, n_loops=10, verbose=False, model='multinomial',
            CC=False, check_convergence=False,
            kernel=['rbf', 'rbf', 'rbf'],
            kernel_params = [{'gamma': gmma_list[i]} for i in range(3)],
            condition=True)
est.fit(XX_train, y_train)
cooks = est.cooks_distance()
out = np.argmax(cooks)
print('Outlier at index {}'.format(out))
for i in range(y_train.shape[0]):
    if i != out:
        plt.plot([i,i], [0,cooks[i]], c='grey')
    else:
        continue
plt.plot([out,out], [0,cooks[out]], c='red')
plt.xlabel('Index')
plt.ylabel("Cook's distance")
if tikz:
    tikzplotlib.save("cooks.tex")
else:
    plt.show()


