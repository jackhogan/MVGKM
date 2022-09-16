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

# need to wrap around MVGLM so that we can automatically calculate gamma

class custom_transform(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scl = StandardScaler()
    
    def fit(self, X, y=None):
        self.views_ind = X.views_ind
        self.scl.fit(X)
        return self
    
    def transform(self, X, y=None):
        X_scl = self.scl.transform(X)
        XX = MultiModalArray(X_scl, views_ind = self.views_ind)
        return XX

# define new estimator, which is just 3 SVMs that combine their predictions
class custom_learner(BaseEstimator, TransformerMixin):
    def __init__(self, C1=1, C2=1, C3=1, gmma=None):
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.gmma = gmma
        self.est1 = SVC(C=C1, kernel='rbf', gamma=gmma)
        self.est2 = SVC(C=C2, kernel='rbf', gamma=gmma)
        self.est3 = SVC(C=C3, kernel='rbf', gamma=gmma)
        
    def fit(self, X, y=None):
        if self.gmma == None:
            self.gmma = [1/(2*np.mean(
                            metrics.pairwise_distances(X.get_view(i))
                            )**2) for i in range(3)]
            self.est1 = SVC(C=self.C1, kernel='rbf', gamma=self.gmma[0])
            self.est2 = SVC(C=self.C2, kernel='rbf', gamma=self.gmma[1])
            self.est3 = SVC(C=self.C3, kernel='rbf', gamma=self.gmma[2])
        self.est1.fit(X.get_view(0), y)
        self.est2.fit(X.get_view(1), y)
        self.est3.fit(X.get_view(2), y)
        return self
    
    def transform(self, X, y=None):
        pass
    
    def predict(self, X, y=None):
        pred_mat = np.zeros((X.get_view(0).shape[0],3))
        pred_mat[:,0] = self.est1.predict(X.get_view(0))
        pred_mat[:,1] = self.est2.predict(X.get_view(1))
        pred_mat[:,2] = self.est3.predict(X.get_view(2))
        return pred_mat.mean(axis=1)

# construct pipeline
Pipe = Pipeline([
    ('scale', custom_transform()),
    ('clf', custom_learner())
])

# set grid of parameters to search over
params = {
        'clf__C1': np.geomspace(1e-4, 1e3, 20),
        'clf__C2': np.geomspace(1e-4, 1e3, 20),
        'clf__C3': np.geomspace(1e-4, 1e3, 20)
        }

scoring = metrics.make_scorer(custom_scorer.lambda_score, 
                                         greater_is_better=False)

if __name__ == "__main__":
    grid = GridSearchCV(Pipe, params, cv=5,
                            verbose=10, n_jobs=-1, scoring=scoring)
    grid.fit(XX, y_train)

    print('SVM late best score: {}'.format(grid.best_score_))

    # save the grid results
    joblib.dump(grid, '../results/svm_late.pkl')

