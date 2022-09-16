# load required modules
import os, sys
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
# add parent directory to path
dir1 = os.path.dirname(os.path.abspath('..'))
if not dir1 in sys.path: sys.path.append(dir1)
from mvglm_algo import MVGLM
from multimodal.datasets.data_sample import MultiModalArray
from multimodal.kernels.mvml import MVML
import custom_scorer

# load data and create train/test split
# load in the training and testing data
with open('../data/audio_feats.pkl', 'rb') as f:
    all_data = pickle.load(f)
with open('../data/audio_labs.pkl', 'rb') as f:
    labels = pickle.load(f)

XX_full = MultiModalArray(all_data)

XX, XX_test, y_train, y_test = train_test_split(XX_full, labels, test_size=0.8,
                                                random_state=123)
# add small amount of noise to response so that MVML sees it as a float
y_train += np.random.RandomState(123).normal(scale=1e-8, size=y_train.shape)

custom_transform = None
class custom_learner(BaseEstimator, TransformerMixin):
    def __init__(self, lmbda=1, eta=1, gmma=None):
        self.lmbda = lmbda
        self.eta = eta
        self.gmma = gmma
        self.est = MVML(lmbda=lmbda, eta=eta, nystrom_param=1.0,
                        kernel=['rbf', 'rbf', 'rbf'],
                        learn_A=1, learn_w=1, kernel_params=None
                    )
        
    def fit(self, X, y=None):
        if self.gmma == None:
            self.gmma = [1/(2*np.mean(
                            metrics.pairwise_distances(X.get_view(i))
                            )**2) for i in range(3)]
            self.est.kernel_params = [{'gamma':self.gmma[i]} for i in range(3)] 
        self.est.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        pass
    
    def predict(self, X, y=None):
        p = self.est.predict(X)
        return(p)

# construct pipeline
Pipe = Pipeline([
    ('clf', custom_learner())
])

# set grid of parameters to search over
params = {
        'clf__lmbda': np.geomspace(1e-4, 1e3, 20),
        'clf__eta': np.geomspace(1e-4, 1e3, 20)
        }

if __name__ == "__main__":
    grid = GridSearchCV(Pipe, params, cv=5, verbose=10, n_jobs=-1, 
            scoring=metrics.make_scorer(custom_scorer.lambda_score,
                                        greater_is_better=False)
            )
    grid.fit(XX, y_train)

    print('MVML best score: {}'.format(grid.best_score_))

    # save the grid results
    joblib.dump(grid, '../results/mvml.pkl')

