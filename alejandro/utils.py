
from sklearn.base import BaseEstimator, TransformerMixin
from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pickle

class DummyClassifier:
    def __init__(self, n=100):
        self.n = n
        self.classes_ = None
        self.probabilities = None
        self.sample_output = None
    def fit(self, X, y):
        aux = (y.value_counts()/y.count()).sort_index()
        self.classes_ = np.array(aux.index)
        self.probabilities = np.array(aux)
        del aux
        self.sample_output = self.classes_[np.flip(self.probabilities.argsort())]
        
    def __predict_v1(self, X):
        return np.array([list(self.sample_output[:self.n])]*X.shape[0])

    def __predict_v2(self, X):
        out = []
        for x in X:
            aux_row = tuple()
            for c in self.sample_output:
                if c not in x:
                    aux_row += (c,)
                if len(aux_row) == 100:
                    break
            out.append(aux_row)
        return np.array(out)
    def predict(self, X):
        return __predict_v1(self,X)

    def pred_scores(self, y_true, y_pred):
        scores = []
        for _y_true, _y_pred in zip(y_true, y_pred):
            index=np.where(_y_pred == _y_true)[0]
            if not index:
                index = 0
            else:
                index = index[0]
            scores.append((index/self.n))
        return scores

    def score(self, y_true, y_pred):
        return np.mean(pred_scores(self, y_true, y_pred))
    
class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Wraps `MultiLabelBinarizer` in a form that can work with `ColumnTransformer`. Note
    that input X has to be a `pandas.DataFrame`.
    """
    def __init__(self):
        self.mlbs = list()
        self.n_columns = 0
        self.categories_ = self.classes_ = list()

    def fit(self, X:DataFrame, y=None):
        for i in range(X.shape[1]): # X can be of multiple columns
            mlb = MultiLabelBinarizer()
            mlb.fit(X.iloc[:,i])
            self.mlbs.append(mlb)
            self.classes_.append(mlb.classes_)
            self.n_columns += 1
        return self

    def transform(self, X: DataFrame):
        if self.n_columns == 0:
            raise ValueError('Please fit the transformer first.')
        if self.n_columns != X.shape[1]:
            raise ValueError(f'The fit transformer deals with {self.n_columns} columns '
                             f'while the input has {X.shape[1]}.'
                            )
        result = list()
        for i in range(self.n_columns):
            result.append(self.mlbs[i].transform(X.iloc[:,i]))

        result = np.concatenate(result, axis=1)
        return result

def load_model(pkl_filename):
    try:
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
        return pickle_model
    except FileNotFoundError as e:
        return
    except Exception as e:
        print(e)
        return
    
def save_model(model, pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    return 