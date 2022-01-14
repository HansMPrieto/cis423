import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# This class maps values in a column, numeric or categorical.
class MappingTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, mapping_column, mapping_dict:dict):  
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
   

class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=True):  
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first
  
  #fill in the rest below
  def fit(self, X, y = None):
    print("Warning: OHETransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'OHETransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'OHETransformer.transform unknown column {self.target_column}'
    X_ = pd.get_dummies(X,
                          prefix=self.target_column,    #your choice
                          prefix_sep='_',     #your choice
                          columns=[self.target_column],
                          dummy_na=self.dummy_na,    #will try to impute later so leave NaNs in place
                          drop_first=self.drop_first    #will drop Belfast and infer it
                          )
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
    

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
    assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
    self.column_list = column_list
    self.action = action

  #fill in rest below
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
    assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
    self.column_list = column_list
    self.action = action

  #fill in rest below
  def fit(self, X, y = None):
    print("Warning: DropColumnsTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'DropColumnsTransformer.transform expected Dataframe but got {type(X)} instead.'
    temp_list = list(set(self.column_list) - set(X.columns.to_list()))
    assert len(temp_list) == 0, f"{temp_list} not in table"

    X_ = X.drop(columns=self.column_list) if self.action == 'drop' else X[self.column_list]
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result


class PearsonTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, threshold):
    self.threshold = threshold

  #define methods below
  def fit(self, X, y = None):
    print("Warning: PearsonTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'PearsonTransformer.transform expected Dataframe but got {type(X)} instead.'
    df_corr = X.corr(method='pearson')
    masked_df = df_corr.abs() > self.threshold
    upper_mask = np.triu(masked_df, 1)
    correlated_columns = [col for (index, col) in enumerate(masked_df) if np.any(upper_mask[:,index])]
    X_ = X.drop(columns=correlated_columns)
    return X_
  
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
