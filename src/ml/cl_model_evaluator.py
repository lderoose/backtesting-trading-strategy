#!/usr/bin/python3.8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score


class ModelEvaluator:
        
    @staticmethod
    @njit()
    def __proba_to_integer(probas, thresh):
        array = np.empty(shape=len(probas), dtype=np.uint64)
        for i, p in enumerate(probas):
            if p > thresh:
                array[i] = 1
            else:
                array[i] = 0
        return array
    
    @staticmethod
    def _convert(proba1_pred, lst_thresh):
        """
        convert all probas to integer for many threshold.
        """
        array = np.empty(shape=(len(proba1_pred), len(lst_thresh)), dtype=np.uint64)
        for i, thresh in enumerate(lst_thresh):
            array[:, i] = __class__.__proba_to_integer(proba1_pred, thresh)
        return array
    
    @staticmethod
    def compute_metrics(proba1_pred, labels, lst_thresh=np.arange(0, 1.01, .01), beta=1):
        """
        given a vector of probabilities compute accuracy, precision score, recall score and fbeta score for each threshold.
        Parameters
        ----------
        proba1_pred : numpy.ndarray
            vector of probabilities
        labels : numpy.ndarray
            true integers
        lst_thresh : numpy.ndarray or list, optional
            all the threshold, by default np.arange(0, 1.01, .01)
        beta : int, optional
            parameter for fbeta score, by default 1
        Returns
        -------
        pandas.DataFrame
            metric for each threshold.
        """
        lst_metrics = ["accuracy", "precision", "recall", "fbeta_score"]
        array_metrics = np.empty(shape=(len(lst_thresh), len(lst_metrics)), dtype=np.float64)
        array = __class__._convert(proba1_pred, lst_thresh)
        for i in tqdm(range(len(lst_thresh))):               
            if np.array_equal(array[:,i], array[:,i-1]):
                array_metrics[i,:] = array_metrics[i-1,:]
            else:
                array_metrics[i,0] = accuracy_score(labels, array[:,i])
                array_metrics[i,1] = precision_score(labels, array[:,i])
                array_metrics[i,2] = recall_score(labels, array[:,i])
                array_metrics[i,3] = fbeta_score(labels, array[:,i], beta)
            
        df_scores = pd.DataFrame(array_metrics, index=lst_thresh, columns=lst_metrics)
        return df_scores
    
    @staticmethod
    def best_fbeta_score(tab_metrics):
        """
        threshold who maximizes fbeta score.
        """
        thresh = tab_metrics.index[np.argmax(tab_metrics['fbeta_score'])]
        return thresh
    
    @staticmethod
    def plot_metrics(df_scores, lst_thresh=np.arange(0, 1.05, .05), figsize_x=20, figsize_y=5):
        """
        display computed metric for each threshold.
        """
        plt.figure(figsize=(figsize_x, figsize_y))
        for col in df_scores.columns:
            plt.plot(df_scores[col], label=col)
        plt.xticks(lst_thresh)
        plt.title("metric evolution model")
        plt.legend();
