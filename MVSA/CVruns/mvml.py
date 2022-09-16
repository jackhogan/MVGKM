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

# unpack data and labels
with open('../data/image_mat.pkl', 'rb') as f:
    img_mat = pickle.load(f)

with open('../data/texts.pkl', 'rb') as f:
    texts = pickle.load(f)

with open('../data/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
labels = labels.astype(int)

# split into training and testing sets
texts_train, texts_test, \
y_train, y_test, \
img_train, img_test = train_test_split(texts, labels, img_mat,
                                       test_size=0.5, random_state=123)

XX = pd.DataFrame({'texts': texts_train, 
                   'imgs': img_train.tolist()
                  })
XX_test = pd.DataFrame({'texts': texts_test, 
                   'imgs': img_test.tolist()
                  })

#############################################################
## MVGLM Grid Search
#############################################################

# define new transform function
class custom_transform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        text_df = X['texts'].tolist()
        self.count_vect = CountVectorizer()
        self.count_vect.fit(text_df)
        return self
    
    def transform(self, X, y=None):
        text_df = X['texts'].tolist()
        img_df = np.array(X['imgs'].values.tolist())
        X_text = self.count_vect.transform(text_df).toarray()
        X_dict = {0: X_text, 1: img_df}
        return MultiModalArray(X_dict)

# need to wrap around MVGLM so that we can automatically calculate gamma
class custom_learner(BaseEstimator, TransformerMixin):
    def __init__(self, lmbda=1, eta=1, gmma=None):
        self.lmbda = lmbda
        self.eta = eta
        self.gmma = gmma
        self.est = MVML(lmbda=lmbda, eta=eta, nystrom_param=1.0, learn_A=1,
                         learn_w=1, n_loops=6, 
                         kernel=['linear', 'rbf'], kernel_params = None)
        
    def fit(self, X, y=None):
        if self.gmma == None:
            self.gmma = 1/(2*np.mean(
                                metrics.pairwise_distances(X.get_view(1))
                                )**2)
            self.est.kernel_params = [{}, {'gamma': self.gmma}] 
        self.est.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        pass
    
    def predict(self, X, y=None):
        p = self.est.predict(X)
        return(p)

# construct pipeline
Pipe = Pipeline([
    ('transform', custom_transform()),
    ('clf', custom_learner())
])

# set grid of parameters to search over
params = {
        'clf__lmbda': np.geomspace(1e-4, 1e3, 20),
        'clf__eta': np.geomspace(1e-4, 1e3, 20),
        }

if __name__ == "__main__":
    # instantiate the grid search instance
    grid = GridSearchCV(Pipe, params, cv=5,
                        verbose=10, n_jobs=-1, scoring='accuracy')
    grid.fit(XX, y_train)

    print('MVML best score: {}'.format(grid.best_score_))

    # save the grid results
    joblib.dump(grid, '../results/mvml.pkl')

