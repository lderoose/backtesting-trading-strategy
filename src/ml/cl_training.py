#!/usr/bin/python3.8


import numpy as np
import pandas as pd
from tqdm import tqdm

from cl_features_importances import FeatureImportances


class WindowTrainingSpec:
    """
    relevant to use WindowTrainingSpec instead WindowTraining if labels are computed with cl_labelling.Labelling.
    """
    
    def __init__(self, size_window, refit_fqz):
        if isinstance(size_window, int):
            if size_window > 1:
                self.size_window = size_window
            else:
                raise ValueError("parameter size_window must be greater than 1.")
        else:
            raise TypeError(f"expected type int for size_window parameter, get {type(size_window)} instead.")
            
        if isinstance(refit_fqz, int):
            if refit_fqz > 0:
                self.refit_fqz = refit_fqz
            else:
                raise ValueError("parameter refit_fqz must be greater than 0.")
        else:
            raise TypeError(f"expected type int for refit_fqz parameter, get {type(refit_fqz)} instead.")
        
    def _compute_lag(self, idx, index_knowledge):
        for lag in range(1, idx, 1):
            b_inf = max(0, idx - lag - self.size_window)
            vector = index_knowledge[b_inf:(idx-lag)]
            if np.max(vector) <= idx:
                break
        return lag
            
    def fit_predict(self, model, X, y, index_knowledge, sample_weight=None, fit_or_refit="fit", predict_proba=True, accumulate_window=False,
                    get_feat_imp=False):
        """
        1 - fit (or refit) a model on last n data,
        2 - predict until next fitting
        3 - fit (or refit) a model on last n data
        4 - ...
        Parameters
        ----------
        model : supported model
            must have a .fit method
        X : pandas.DataFrame
            features
        y : numpy.ndarray
            labels
        index_knowledge : numpy.ndarray
            at what time label can be guess
        sample_weight : numpy.ndarray, optional
            vector of weights for each row, by default None means equal for each rows.
        fit_or_refit : str, optional
            fit if next predictions will be make with a new model or refit if next predictions will be make with the same model fitted
            on new data, by default "fit".
        predict_proba : bool, optional
            by default True
        accumulate_window : bool, optional
            new data = old_data + new_data ? or new_data = new_data, by default False
        get_feat_imp : bool, optional
            provide dict of features importances (sum of abs(importances) of each model), by default False
        Returns
        -------
        pandas.DataFrame or pandas.DataFrame and dict (if get_feat_imp == True)
            predictions
        Raises
        ------
        ValueError
            fit_or_refit != 'fit' or 'refit'.
        """
        if get_feat_imp:
            dic_imp = {}
        if predict_proba:
            preds = np.empty(shape=(len(X)-self.size_window, 2), dtype=np.float64)
        else:
            preds = np.empty(shape=len(X)-self.size_window, dtype=np.float64)
        
        if sample_weight is None:
            sample_weight = np.ones(shape=len(X), dtype=np.uint64)
        
        for i, j in enumerate(tqdm(range(self.size_window, len(X), self.refit_fqz))):
            
            lag = self._compute_lag(j, index_knowledge)  
            
            if fit_or_refit == "fit":  
                
                model_unfit = model
                if accumulate_window:
                    model_fitted = model_unfit.fit(
                        X[:(j-lag)],
                        y[:(j-lag)],
                        sample_weight[:(j-lag)]
                    )
                else:
                    b_inf = max(0, j - lag - self.size_window)
                    model_fitted = model_unfit.fit(
                        X[b_inf:(j-lag)],
                        y[b_inf:(j-lag)],
                        sample_weight[b_inf:(j-lag)]
                    )
                    
            elif fit_or_refit == "refit":
                if i == 0:
                    model_fitted = model
                    
                if accumulate_window:
                    model_fitted = model_fitted.fit(
                        X[:(j-lag)],
                        y[:(j-lag)],
                        sample_weight[:(j-lag)]
                    )
                else:
                    b_inf = max(0, j - lag - self.size_window)
                    model_fitted = model_fitted.fit(
                        X[b_inf:(j-lag)],
                        y[b_inf:(j-lag)],
                        sample_weight[b_inf:(j-lag)]
                    )
            else:
                raise ValueError("parameter fit_or_refit must be 'fit' or 'refit'.")
                    
            start = i*self.refit_fqz
            stop = self.refit_fqz + i*self.refit_fqz
            
            if get_feat_imp:
                dic_imp_tmp = FeatureImportances.compute(model_fitted, X.columns.tolist(), reverse=False, mode_abs=True)
                if dic_imp:
                    dic_imp = FeatureImportances.accumulate_dic_imp(dic_imp, dic_imp_tmp)
                else:
                    dic_imp = dic_imp_tmp

            if predict_proba:
                preds[start:stop, :] = model_fitted.predict_proba(X[j:j+self.refit_fqz])
            else:
                preds[start:stop] = model_fitted.predict(X[j:j+self.refit_fqz])
        
        preds = pd.DataFrame(preds, index=X.index[self.size_window:])    
        if get_feat_imp:
            dic_imp = dict(sorted(dic_imp.items(), key=lambda x: x[1], reverse=True))  # sort by values
            return preds, dic_imp
        else:
            return preds
        
class WindowTraining:
    
    def __init__(self, size_window, refit_fqz):       
        if isinstance(size_window, int):
            if size_window > 1:
                self.size_window = size_window
            else:
                raise ValueError("parameter size_window must be greater than 1.")
        else:
            raise TypeError(f"expected type int for size_window parameter, get {type(size_window)} instead.")
            
        if isinstance(refit_fqz, int):
            if refit_fqz > 0:
                self.refit_fqz = refit_fqz
            else:
                raise ValueError("parameter refit_fqz must be greater than 0.")
        else:
            raise TypeError(f"expected type int for refit_fqz parameter, get {type(refit_fqz)} instead.")
            
    def fit_predict(self, model, X, y, sample_weight=None, fit_or_refit="fit", predict_proba=True, accumulate_window=False,
                    get_feat_imp=False):
        """
        1 - fit (or refit) a model on last n data,
        2 - predict until next fitting
        3 - fit (or refit) a model on last n data
        4 - ...
        Parameters
        ----------
        model : supported model
            must have a .fit method
        X : pandas.DataFrame
            features
        y : numpy.ndarray
            labels
        sample_weight : numpy.ndarray, optional
            vector of weights for each row, by default None means equal for each rows.
        fit_or_refit : str, optional
            fit if next predictions will be make with a new model or refit if next predictions will be make with the same model fitted
            on new data, by default "fit".
        predict_proba : bool, optional
            by default True
        accumulate_window : bool, optional
            new data = old_data + new_data ? or new_data = new_data, by default False
        get_feat_imp : bool, optional
            provide dict of features importances (sum of abs(importances) of each model), by default False
        Returns
        -------
        pandas.DataFrame or pandas.DataFrame and dict (if get_feat_imp == True)
            predictions
        Raises
        ------
        ValueError
            fit_or_refit != 'fit' or 'refit'.
        """
        if get_feat_imp:
            dic_imp = {}
        if predict_proba:
            preds = np.empty(shape=(len(X)-self.size_window, 2), dtype=np.float64)
        else:
            preds = np.empty(shape=len(X)-self.size_window, dtype=np.float64)
        
        if sample_weight is None:
            sample_weight = np.ones(shape=len(X), dtype=np.uint64)
        
        for i, j in enumerate(tqdm(range(self.size_window, len(X), self.refit_fqz))):
            
            if fit_or_refit == "fit":  
                                
                model_unfit = model
                if accumulate_window:
                    model_fitted = model_unfit.fit(
                        X[:j],
                        y[:j],
                        sample_weight[:j]
                    )
                else:
                    model_fitted = model_unfit.fit(
                        X[j-self.size_window:j],
                        y[j-self.size_window:j],
                        sample_weight[j-self.size_window:j]
                    )
                    
            elif fit_or_refit == "refit":
                if i == 0:
                    model_fitted = model
                    
                if accumulate_window:
                    model_fitted = model_fitted.fit(
                        X[:j],
                        y[:j],
                        sample_weight[:j]
                    )
                else:
                    model_fitted = model_fitted.fit(
                        X[j-self.size_window:j],
                        y[j-self.size_window:j],
                        sample_weight[j-self.size_window:j]
                    )
            else:
                raise ValueError("parameter fit_or_refit must be 'fit' or 'refit'.")
                    
            start = i*self.refit_fqz
            stop = self.refit_fqz + i*self.refit_fqz
            
            if get_feat_imp:
                dic_imp_tmp = FeatureImportances.compute(model_fitted, X.columns.tolist(), reverse=False, mode_abs=True)
                if dic_imp:
                    dic_imp = FeatureImportances.accumulate_dic_imp(dic_imp, dic_imp_tmp)
                else:
                    dic_imp = dic_imp_tmp

            if predict_proba:
                preds[start:stop, :] = model_fitted.predict_proba(X[j:j+self.refit_fqz])
            else:
                preds[start:stop] = model_fitted.predict(X[j:j+self.refit_fqz])
        
        preds = pd.DataFrame(preds, index=X.index[self.size_window:])  
        if get_feat_imp:
            dic_imp = dict(sorted(dic_imp.items(), key=lambda x: x[1], reverse=True))  # sort by values
            return preds, dic_imp
        else:
            return preds
