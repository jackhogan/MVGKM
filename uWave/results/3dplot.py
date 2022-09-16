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
from mpl_toolkits import mplot3d

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

ax = plt.axes(projection='3d')
for s in np.argwhere(y_train == 4).squeeze()[:50]:
# Data for a three-dimensional line
    zline = XX_train.get_view(0)[s,:].T
    xline = XX_train.get_view(1)[s,:].T
    yline = XX_train.get_view(2)[s,:].T
    ax.plot3D(xline, yline, zline, alpha=0.4)
    
for s in [637]:
    zline = XX_train.get_view(0)[s,:].T
    xline = XX_train.get_view(1)[s,:].T
    yline = XX_train.get_view(2)[s,:].T
    ax.plot3D(xline, yline, zline, 'red')
plt.show()
