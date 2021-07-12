#!/usr/bin/python3.8


from collections import Counter

import matplotlib.pyplot as plt
from lightgbm.sklearn import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class FeatureImportances:
    
    @staticmethod
    def _get_importances(model):
        """
        get features importances depending on the model.
        Parameters
        ----------
        model : see supported type
        Returns
        -------
        numpy.ndarray
            features importances from model API.
        Raises
        ------
        ValueError
            if type of model is not support.
        """
        if isinstance(model, LogisticRegression):
            importances = model.coef_
        elif isinstance(model, RandomForestClassifier):
            importances = model.feature_importances_
        elif isinstance(model, LGBMClassifier):
            importances = model.feature_importances_
        else:
            raise ValueError(f"specify this kind of model : {type(model)}")
        return importances.ravel()
    
    @staticmethod
    def accumulate_dic_imp(dic_imp_last, dic_imp_current):
        """
        add values from a dict to an another.
        """
        counter_last = Counter(dic_imp_last)
        counter_current = Counter(dic_imp_current)
        counter_sum = counter_last + counter_current
        return dict(counter_sum)
    
    @staticmethod
    def compute(model, features, reverse=True, mode_abs=False):
        """
        compute feature importances and store into python dict.
        key: feature name
        values: importances
        """
        dic_imp = {}
        importances = __class__._get_importances(model)
        for feature, importance in zip(features, importances):
            if mode_abs:
                dic_imp[feature] = abs(importance)
            else:
                dic_imp[feature] = importance
        dic_imp = dict(sorted(dic_imp.items(), key=lambda x: x[1], reverse=reverse))  # sort by values
        return dic_imp
    
    @staticmethod
    def best_n(n=1, dic_imp=None, model=None, features=None, include_score=False):
        """
        get list of best features regarding their importances given model or dic_imp.
        Parameters
        ----------
        n : int, optional
            best 'n' features, by default 1
        dic_imp : dict, optional
            return from compute method, by default None
        model : see supported model, optional
            by default None
        features : list, optional
            the name of the features, by default None
        include_score : bool, optional
            if True, return a list of tuples which contain (feature, importance) else return best features only, by default False
        Returns
        -------
        list
            list of str if include_score is True else list of tuples.
        """
        assert n > 0, 'n must be > 0'
        assert (dic_imp is not None) | ((model is not None) & (features is not None)), "dic_imp must not be None OR model AND features must not be None"
        if not dic_imp:
            dic_imp = __class__.compute(model, features)
        
        lst_best_n_features = []
        if include_score:
            for i, (feature, importance) in enumerate(dic_imp.items()):
                lst_best_n_features.append((feature, importance))
                if i+1 >= n:
                    break
        else:
            for i, feature in enumerate(dic_imp.keys()):
                lst_best_n_features.append(feature)
                if i+1 >= n:
                    break
        return lst_best_n_features
    
    @staticmethod
    def worst_n(n=1, dic_imp=None, model=None, features=None, include_score=False):
        """
        get list of worst features regarding their importances given model or dic_imp.
        Parameters
        ----------
        n : int, optional
            worst 'n' features, by default 1
        dic_imp : dict, optional
            return from compute method, by default None
        model : see supported model, optional
            by default None
        features : list, optional
            the name of the features, by default None
        include_score : bool, optional
            if True, return a list of tuples which contain (feature, importance) else return best features only, by default False
        Returns
        -------
        list
            list of str if include_score is True else list of tuples.
        """
        assert n > 0, 'n must be > 0'
        assert (dic_imp is not None) | ((model is not None) & (features is not None)), "dic_imp must not be None OR model AND features must not be None"
        if not dic_imp:
            dic_imp = __class__.compute(model, features, False)
        else:
            dic_imp = dict(sorted(dic_imp.items(), key=lambda x: x[1], reverse=False))
        
        lst_worst_n_features = []
        if include_score:
            for i, (feature, importance) in enumerate(dic_imp.items()):
                lst_worst_n_features.append((feature, importance))
                if i+1 >= n:
                    break
        else:
            for i, feature in enumerate(dic_imp.keys()):
                lst_worst_n_features.append(feature)
                if i+1 >= n:
                    break
        return lst_worst_n_features
    
    @staticmethod
    def sift(thresh=0, dic_imp=None, model=None, features=None, include_score=False):
        """
        compute importances and filter features that are not relevant regarding threshold.
        Parameters
        ----------
        thresh : int, optional
            by default 0
        dic_imp : dict, optional
            return from compute method, by default None
        model : see supported model, optional
            by default None
        features : list, optional
            the name of the features, by default None
        include_score : bool, optional
            if True, return a list of tuples which contain (feature, importance) else return best features only, by default False
        Returns
        -------
        list
            features where importances are greater than threshold.
        """
        assert (dic_imp is not None) | ((model is not None) & (features is not None)), "dic_imp must not be None OR model AND features must not be None"
        if not dic_imp:
            dic_imp = __class__.compute(model, features, False)
        else:
            dic_imp = dict(sorted(dic_imp.items(), key=lambda x: x[1], reverse=False))
        
        lst_filtered_features = []
        for feature, importance in dic_imp.items():
            if (thresh >= importance) & (-thresh <= importance):
                if include_score:
                    lst_filtered_features.append((feature, importance))
                else:
                    lst_filtered_features.append(feature)
        return lst_filtered_features
        
    @staticmethod
    def plot(dic_imp):
        """
        display features importances.
        """
        plt.figure(figsize=(len(dic_imp)*5, 5))
        plt.bar(dic_imp.keys(), dic_imp.values())
        plt.show()
        return None
