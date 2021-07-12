#!/usr/bin/python3.8


import numpy as np
from numba import njit, types, typed


"""
    Given parameters and predictions from both models, simulate many portfolio evolution.
    
        m1 <=> model stable_to_crypto
        m2 <=> model crypto_to_stable
        position 0 <=> long stable
        position 1 <=> long crypto
        action 0 <=> do nothing
        action 1 <=> move position
        
        Numba 0.53.0 : 
            - doesn't support python object of collections.deque().
            - doesn't support str(float), repr(float), format(float, 'f'), numpy.format_float_positional(float).
            - numba.typed.Dict() must contain object of the same type.
        
        NB : performances between python and numba cannot be observed by only removing 'njit' decorator because looping on numba list take more time than on python list.
             assume stable coin have a fix return (0%) at each period.
             some parts of the code could be avoided to reduce the size of the script (for example methods _python_params_to_numba_) but they have been kept for 
             readability.
"""

class StrategySimulator:
    
    START_POSITION = 1
    LST_STRATEGIES = [
        "traditional",
        "combine_both_models",
        "moving_average_signal",
        "over_limit_signal",
        "consecutive_streak_signal"
    ]
    
    def __init__(self, K, F, rates):
        """
        Parameters
        ----------
        K : int or float
            starting capital
        F : float
            transaction fees, not in %. Means for example 0.1% of transaction fees: F = 0.001.
        rates : pandas.Series
            interest rate, index must be in accordance with predictions
        """
        self.K = K
        self.F = F
        self.rates = rates
    
    def _check_strategy(self, dic):
        """
        check if the dict contains an allowed strategy name.
        Parameters
        ----------
        dic : dict
            dict of parameters
        Returns
        -------
        str
            strategy name
        Raises
        ------
        ValueError
        """
        strategy = list(dic.keys())[0]
        if strategy in self.LST_STRATEGIES:
            return strategy
        else:
            raise ValueError(f"unknown strategy: '{strategy}'. Choose a strategy among {', '.join(self.LST_STRATEGIES)}.")
            
    @staticmethod
    def _get_n_sims(dic):
        """
        compute the number of simulations that will be done regarding dict of parameters.
        Parameters
        ----------
        dic : dict
            dict of parameters
        Returns
        -------
        int
            number of simulations
        """
        n_sims = 1
        for v in dic.values():
            if isinstance(v, list) | isinstance(v, np.ndarray):
                n_sims *= len(v)
        return n_sims
    
    @staticmethod
    def __convert_python_list_to_numba_list(obj):
        """
        convert python list to numba list.
        Parameters
        ----------
        obj : list or numpy.ndarray
        Returns
        -------
        numba.typed.List
        """
        assert isinstance(obj, list) | isinstance(obj, np.ndarray), f"{obj} is not list or numpy ndarray."
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
        numba_lst = typed.List(obj)
        return numba_lst
    
    @staticmethod
    def __cast_python_to_numba_list(obj):
        """
        convert a python object to numba list.
        Parameters
        ----------
        obj : float, int, list or numpy.ndarray
        Returns
        -------
        numba.typed.List
        Raises
        ------
        TypeError
        """
        if isinstance(obj, float) | isinstance(obj, int):
            lst = [obj]
        elif isinstance(obj, list):
            lst = obj.copy()
        elif isinstance(obj, np.ndarray):
            lst = obj.tolist()
        else:
            raise TypeError(f"expected type float, int, list or numpy. Get {type(obj)} instead.")
        numba_lst = typed.List(lst)
        return numba_lst
    
    @staticmethod
    def _python_params_to_numba_traditional(dic):
        """
        convert parameters belongs to dict (python dict) to many supported numba 0.53.0 object for strategy 'traditional'.
        """
        numba_lst_thresh_stable_to_crypto = __class__.__convert_python_list_to_numba_list(dic["thresh_m1"])
        numba_lst_thresh_crypto_to_stable = __class__.__convert_python_list_to_numba_list(dic["thresh_m2"])
        return numba_lst_thresh_stable_to_crypto, numba_lst_thresh_crypto_to_stable
    
    @staticmethod
    def _python_params_to_numba_cbm(dic):
        """
        convert parameters belongs to dict (python dict) to many supported numba 0.53.0 object for strategy 'combine both models'.
        """
        numba_lst_thresh_stable_to_crypto = __class__.__convert_python_list_to_numba_list(dic["thresh_m1"])
        numba_lst_thresh_crypto_to_stable = __class__.__convert_python_list_to_numba_list(dic["thresh_m2"])
        conflict_resolution = dic["conflict_resolution"]
        return numba_lst_thresh_stable_to_crypto, numba_lst_thresh_crypto_to_stable, conflict_resolution
    
    @staticmethod
    def _python_params_to_numba_mas(dic):
        """
        convert parameters belongs to dict (python dict) to many supported numba 0.53.0 object for strategy 'moving average signal'.
        """
        numba_lst_thresh_stable_to_crypto = __class__.__convert_python_list_to_numba_list(dic["thresh_m1"])
        numba_lst_thresh_crypto_to_stable = __class__.__convert_python_list_to_numba_list(dic["thresh_m2"])
        numba_lst_window_stable_to_crypto = __class__.__cast_python_to_numba_list(dic["window_stable_to_crypto"])
        numba_lst_window_crypto_to_stable = __class__.__cast_python_to_numba_list(dic["window_crypto_to_stable"])
        return numba_lst_thresh_stable_to_crypto, numba_lst_thresh_crypto_to_stable, numba_lst_window_stable_to_crypto, numba_lst_window_crypto_to_stable
    
    @staticmethod
    def _python_params_to_numba_ols(dic):
        """
        convert parameters belongs to dict (python dict) to many supported numba 0.53.0 object for strategy 'over limit signal'.
        """
        numba_lst_thresh_stable_to_crypto = __class__.__convert_python_list_to_numba_list(dic["thresh_m1"])
        numba_lst_thresh_crypto_to_stable = __class__.__convert_python_list_to_numba_list(dic["thresh_m2"])
        numba_lst_limit_stable_to_crypto = __class__.__cast_python_to_numba_list(dic["limit_stable_to_crypto"])
        numba_lst_limit_crypto_to_stable = __class__.__cast_python_to_numba_list(dic["limit_crypto_to_stable"])
        return numba_lst_thresh_stable_to_crypto, numba_lst_thresh_crypto_to_stable, numba_lst_limit_stable_to_crypto, numba_lst_limit_crypto_to_stable
    
    @staticmethod
    def _python_params_to_numba_css(dic):
        """
        convert parameters belongs to dict (python dict) to many supported numba 0.53.0 object for strategy 'cumulative streak signal'.
        """
        numba_lst_thresh_stable_to_crypto = __class__.__convert_python_list_to_numba_list(dic["thresh_m1"])
        numba_lst_thresh_crypto_to_stable = __class__.__convert_python_list_to_numba_list(dic["thresh_m2"])
        numba_lst_consecutive_stable_to_crypto = __class__.__cast_python_to_numba_list(dic["n_cons_stable_to_crypto"])
        numba_lst_consecutive_crypto_to_stable = __class__.__cast_python_to_numba_list(dic["n_cons_crypto_to_stable"])
        return numba_lst_thresh_stable_to_crypto, numba_lst_thresh_crypto_to_stable, numba_lst_consecutive_stable_to_crypto, numba_lst_consecutive_crypto_to_stable
    
    @staticmethod
    def _get_lst_params_strategy_traditional(d_params):
        """
        given dict of parameters, return an ordered list which contain the different parameters used for each simulation.
        """
        lst = []
        if isinstance(d_params["thresh_m1"], list):
            lst_thresh_m1 = d_params["thresh_m1"]
        elif isinstance(d_params["thresh_m1"], np.ndarray):
            lst_thresh_m1 = d_params["thresh_m1"].tolist()
        else:
            lst_thresh_m1 = [d_params["thresh_m1"]]
            
        if isinstance(d_params["thresh_m2"], list):
            lst_thresh_m2 = d_params["thresh_m2"]
        elif isinstance(d_params["thresh_m2"], np.ndarray):
            lst_thresh_m2 = d_params["thresh_m2"].tolist()
        else:
            lst_thresh_m2 = [d_params["thresh_m2"]]
            
        for thresh_m1 in lst_thresh_m1:
            for thresh_m2 in lst_thresh_m2:
                s = f"{thresh_m1:.4f}_{thresh_m2:.4f}"
                lst.append(s)
        return lst
    
    @staticmethod
    def _get_lst_params_strategy_cbm(d_params):
        """
        given dict of parameters, return an ordered list which contain the different parameters used for each simulation.
        """
        lst = []
        if isinstance(d_params["thresh_m1"], list):
            lst_thresh_m1 = d_params["thresh_m1"]
        elif isinstance(d_params["thresh_m1"], np.ndarray):
            lst_thresh_m1 = d_params["thresh_m1"].tolist()
        else:
            lst_thresh_m1 = [d_params["thresh_m1"]]
            
        if isinstance(d_params["thresh_m2"], list):
            lst_thresh_m2 = d_params["thresh_m2"]
        elif isinstance(d_params["thresh_m2"], np.ndarray):
            lst_thresh_m2 = d_params["thresh_m2"].tolist()
        else:
            lst_thresh_m2 = [d_params["thresh_m2"]]
        
        conflict_resolution = d_params["conflict_resolution"]
            
        for thresh_m1 in lst_thresh_m1:
            for thresh_m2 in lst_thresh_m2:
                s = f"{thresh_m1:.4f}_{thresh_m2:.4f}_{conflict_resolution}"
                lst.append(s)
        return lst
    
    @staticmethod
    def _get_lst_params_strategy_mas(d_params):
        """
        given dict of parameters, return an ordered list which contain the different parameters used for each simulation.
        """
        lst = []
        if isinstance(d_params["thresh_m1"], list):
            lst_thresh_m1 = d_params["thresh_m1"]
        elif isinstance(d_params["thresh_m1"], np.ndarray):
            lst_thresh_m1 = d_params["thresh_m1"].tolist()
        else:
            lst_thresh_m1 = [d_params["thresh_m1"]]
            
        if isinstance(d_params["thresh_m2"], list):
            lst_thresh_m2 = d_params["thresh_m2"]
        elif isinstance(d_params["thresh_m2"], np.ndarray):
            lst_thresh_m2 = d_params["thresh_m2"].tolist()
        else:
            lst_thresh_m2 = [d_params["thresh_m2"]]
            
        if isinstance(d_params["window_stable_to_crypto"], list):
            lst_window_1 = d_params["window_stable_to_crypto"]
        elif isinstance(d_params["window_stable_to_crypto"], np.ndarray):
            lst_window_1 = d_params["window_stable_to_crypto"].tolist()
        else:
            lst_window_1 = [d_params["window_stable_to_crypto"]]
            
        if isinstance(d_params["window_crypto_to_stable"], list):
            lst_window_2 = d_params["window_crypto_to_stable"]
        elif isinstance(d_params["window_crypto_to_stable"], np.ndarray):
            lst_window_2 = d_params["window_crypto_to_stable"].tolist()
        else:
            lst_window_2 = [d_params["window_crypto_to_stable"]]
            
        for thresh_m1 in lst_thresh_m1:
            for thresh_m2 in lst_thresh_m2:
                for window_1 in lst_window_1:
                    for window_2 in lst_window_2:
                        s = f"{thresh_m1:.4f}_{thresh_m2:.4f}_{window_1}_{window_2}"
                        lst.append(s)
        return lst
    
    @staticmethod
    def _get_lst_params_strategy_ols(d_params):
        """
        given dict of parameters, return an ordered list which contain the different parameters used for each simulation.
        """
        lst = []
        if isinstance(d_params["thresh_m1"], list):
            lst_thresh_m1 = d_params["thresh_m1"]
        elif isinstance(d_params["thresh_m1"], np.ndarray):
            lst_thresh_m1 = d_params["thresh_m1"].tolist()
        else:
            lst_thresh_m1 = [d_params["thresh_m1"]]
            
        if isinstance(d_params["thresh_m2"], list):
            lst_thresh_m2 = d_params["thresh_m2"]
        elif isinstance(d_params["thresh_m2"], np.ndarray):
            lst_thresh_m2 = d_params["thresh_m2"].tolist()
        else:
            lst_thresh_m2 = [d_params["thresh_m2"]]
            
        if isinstance(d_params["limit_stable_to_crypto"], list):
            lst_limit_1 = d_params["limit_stable_to_crypto"]
        elif isinstance(d_params["limit_stable_to_crypto"], np.ndarray):
            lst_limit_1 = d_params["limit_stable_to_crypto"].tolist()
        else:
            lst_limit_1 = [d_params["limit_stable_to_crypto"]]
            
        if isinstance(d_params["limit_crypto_to_stable"], list):
            lst_limit_2 = d_params["limit_crypto_to_stable"]
        elif isinstance(d_params["limit_crypto_to_stable"], np.ndarray):
            lst_limit_2 = d_params["limit_crypto_to_stable"].tolist()
        else:
            lst_limit_2 = [d_params["limit_crypto_to_stable"]]
            
        for thresh_m1 in lst_thresh_m1:
            for thresh_m2 in lst_thresh_m2:
                for limit_1 in lst_limit_1:
                    for limit_2 in lst_limit_2:
                        s = f"{thresh_m1:.4f}_{thresh_m2:.4f}_{limit_1}_{limit_2}"
                        lst.append(s)
        return lst
    
    @staticmethod
    def _get_lst_params_strategy_css(d_params):
        """
        given dict of parameters, return an ordered list which contain the different parameters used for each simulation.
        """
        lst = []
        if isinstance(d_params["thresh_m1"], list):
            lst_thresh_m1 = d_params["thresh_m1"]
        elif isinstance(d_params["thresh_m1"], np.ndarray):
            lst_thresh_m1 = d_params["thresh_m1"].tolist()
        else:
            lst_thresh_m1 = [d_params["thresh_m1"]]
            
        if isinstance(d_params["thresh_m2"], list):
            lst_thresh_m2 = d_params["thresh_m2"]
        elif isinstance(d_params["thresh_m2"], np.ndarray):
            lst_thresh_m2 = d_params["thresh_m2"].tolist()
        else:
            lst_thresh_m2 = [d_params["thresh_m2"]]
            
        if isinstance(d_params["n_cons_stable_to_crypto"], list):
            lst_n_cons_1 = d_params["n_cons_stable_to_crypto"]
        elif isinstance(d_params["n_cons_stable_to_crypto"], np.ndarray):
            lst_n_cons_1 = d_params["n_cons_stable_to_crypto"].tolist()
        else:
            lst_n_cons_1 = [d_params["n_cons_stable_to_crypto"]]
            
        if isinstance(d_params["n_cons_crypto_to_stable"], list):
            lst_n_cons_2 = d_params["n_cons_crypto_to_stable"]
        elif isinstance(d_params["n_cons_crypto_to_stable"], np.ndarray):
            lst_n_cons_2 = d_params["n_cons_crypto_to_stable"].tolist()
        else:
            lst_n_cons_2 = [d_params["n_cons_crypto_to_stable"]]
            
        for thresh_m1 in lst_thresh_m1:
            for thresh_m2 in lst_thresh_m2:
                for n_cons_1 in lst_n_cons_1:
                    for n_cons_2 in lst_n_cons_2:
                        s = f"{thresh_m1:.2f}_{thresh_m2:.2f}_{n_cons_1}_{n_cons_2}"
                        lst.append(s)
        return lst
    
    @staticmethod
    @njit
    def _apply_strategy_traditional(K, F, START_POSITION, next_rates, n_sims, probas_m1, probas_m2, lst_thresh_stable_to_crypto, lst_thresh_crypto_to_stable):
        """
        *************************************************************** TRADITIONAL STRATEGY EXPLANATION ***************************************************************
        - if at time t, we long crypto, listen predictions from model crypto to stable. Else we long stable, listen predictions from model stable to crypto.
        - if a prediction is > than current threshold, predict 1. Else predict 0.
        ****************************************************************************************************************************************************************
        Parameters
        ----------
        K : int or float
            starting capital
        F : float
            transaction fees, not in %. Means for example 0.1% of transaction fees: F = 0.001.
        START_POSITION : int
            0 if starting on stable, 1 if starting on crypto
        next_rates : numpy.ndarray
            next interest rate
        n_sims : int
            number of simulations
        probas_m1 : numpy.ndarray
            predictions from model crypto_to_stable
        probas_m2 : numpy.ndarray
            predictions from model stable_to_crypto
        lst_thresh_stable_to_crypto : numba.typed.List
            numba list of thresholds for stable_to_crypto, regarding dict of parameters
        lst_thresh_crypto_to_stable : numba.typed.List
            numba list of thresholds for crypto_to_stable, regarding dict of parameters
        """
        def S_traditional(F, p1, p2, next_rate, portfolio_i, position_i, thresh_m1, thresh_m2):
            # NB : in this way, njit decorator is also applied to this function.
            next_portfolio = portfolio_i

            if position_i:
                if p2 > thresh_m2:
                    next_portfolio *= (1 - F)
                    next_position = 0
                    switch = 1
                else:
                    next_portfolio *= (1 + next_rate)
                    next_position = 1
                    switch = 0
                    
            else:
                if p1 > thresh_m1:
                    next_portfolio *= (1 - F)
                    next_portfolio *= (1 + next_rate)
                    next_position = 1
                    switch = 1
                else:
                    next_position = 0
                    switch = 0
            return next_portfolio, next_position, switch
        
        portfolio = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.float64)
        positions = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.uint64)
        switchs = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.uint64)
        portfolio[0, :], positions[0, :], switchs[0, :] = K, START_POSITION, 0
        
        for i, (proba1_m1, proba1_m2, next_rate) in enumerate(zip(probas_m1, probas_m2, next_rates)):
            sim = 0
            for thresh_stable_to_crypto in lst_thresh_stable_to_crypto:
                for thresh_crypto_to_stable in lst_thresh_crypto_to_stable:
                    portfolio[i+1, sim], positions[i+1, sim], switchs[i+1, sim] = S_traditional(
                        F,
                        proba1_m1,
                        proba1_m2,
                        next_rate,
                        portfolio[i, sim],
                        positions[i, sim],
                        thresh_stable_to_crypto,
                        thresh_crypto_to_stable,
                    )
                    sim += 1
        return portfolio, positions, switchs
    
    @staticmethod
    @njit
    def _apply_strategy_cbm(K, F, START_POSITION, next_rates, n_sims, probas_m1, probas_m2, lst_thresh_stable_to_crypto, lst_thresh_crypto_to_stable, conflict_resolution):
        """
        *************************************************************** COMBINE BOTH MODELS STRATEGY EXPLANATION ***************************************************************
        - listen predictions from both models.
        - sometimes, there may occur conflictual case. For example if we long crypto, model stable to crypto can want to stay on stable (predict 0) and model 
          crypto to stable can want to stay on crypto (predict 0). For solving these conflicts many ways are provided by conflict_resolution parameter.
          conflict_resolution parameter:
            - sedentary : in conflictual case, never move from last position (avoid transaction fees)
            - p_max : in conflictual case, listen model which have the most certainty (i.e. the highest probability)
            - stable_priority : in conflictual case, always go to stable.
            - crypto_priority : in conflictual case, always go to crypto.
        ****************************************************************************************************************************************************************
        Parameters
        ----------
        K : int or float
            starting capital
        F : float
            transaction fees, not in %. Means for example 0.1% of transaction fees: F = 0.001.
        START_POSITION : int
            0 if starting on stable, 1 if starting on crypto
        next_rates : numpy.ndarray
            next interest rate
        n_sims : int
            number of simulations
        probas_m1 : numpy.ndarray
            predictions from model crypto_to_stable
        probas_m2 : numpy.ndarray
            predictions from model stable_to_crypto
        lst_thresh_stable_to_crypto : numba.typed.List
            numba list of thresholds for stable_to_crypto, regarding dict of parameters
        lst_thresh_crypto_to_stable : numba.typed.List
            numba list of thresholds for crypto_to_stable, regarding dict of parameters
        conflict_resolution : str
            see explantions
        """
        def S_cbm(F, p1, p2, rate, portfolio_i, position_i, thresh_m1, thresh_m2, conflict_resolution):
            # NB : in this way, njit decorator is also applied to this function.
            next_portfolio = portfolio_i
            next_position, switch = -1, -1  # solve numba.core.errors.UnsupportedError: Failed in nopython mode pipeline (step: nopython frontend) Unsupported op-code encountered: null()
            
            # long crypto
            if position_i:
                # m1 want to stay on stable, m2 want to move to stable
                if (p1 < thresh_m1) & (p2 >= thresh_m2):
                    next_portfolio *= (1 - F)
                    next_position = 0
                    switch = 1

                # conflict case, both want to move
                elif (p1 >= thresh_m1) & (p2 >= thresh_m2):
                    if conflict_resolution == "sedentary":
                        next_portfolio *= (1 + rate)
                        next_position = 1
                        switch = 0

                    elif conflict_resolution == "p_max":
                        if p1 > p2:
                            next_portfolio *= (1 - F)
                            next_position = 0
                            switch = 1
                        else:
                            next_portfolio *= (1 + rate)
                            next_position = 1
                            switch = 0

                    elif conflict_resolution == "stable_priority":
                        next_portfolio *= (1 - F)
                        next_position = 0
                        switch = 1

                    elif conflict_resolution == "crypto_priority":
                        next_portfolio *= (1 + rate)
                        next_position = 1
                        switch = 0

                # conflict case, m1 want to stay on stable, m2 want to stay on crypto
                elif (p1 < thresh_m1) & (p2 < thresh_m2):
                    if conflict_resolution == "sedentary":
                        next_portfolio *= (1 + rate)
                        next_position = 1
                        switch = 0

                    elif conflict_resolution == "p_max":
                        if p1 > p2:
                            next_portfolio *= (1 - F)
                            next_position = 0
                            switch = 1
                        else:
                            next_portfolio *= (1 + rate)
                            next_position = 1
                            switch = 0

                    elif conflict_resolution == "stable_priority":
                        next_portfolio *= (1 - F)
                        next_position = 0
                        switch = 1

                    elif conflict_resolution == "crypto_priority":
                        next_portfolio *= (1 + rate)
                        next_position = 1
                        switch = 0

                # m1 want to move to crypto, m2 want to stay on crypto
                elif (p1 >= thresh_m1) & (p2 < thresh_m2):
                    next_portfolio *= (1 + rate)
                    next_position = 1
                    switch = 0

            # long stable
            else:
                # m1 want to stay on stable, m2 want to move to stable
                if (p1 < thresh_m1) & (p2 >= thresh_m2):
                    next_position = 0
                    switch = 0

                # conflict case, both want to move
                elif (p1 >= thresh_m1) & (p2 >= thresh_m2):
                    if conflict_resolution == "sedentary":
                        next_position = 0
                        switch = 0

                    elif conflict_resolution == "p_max":
                        if p1 > p2:
                            next_position = 0
                            switch = 0
                        else:
                            next_portfolio *= (1 - F)
                            next_portfolio *= (1 + rate)
                            next_position = 1
                            switch = 1

                    elif conflict_resolution == "stable_priority":
                        next_position = 0
                        switch = 0

                    elif conflict_resolution == "crypto_priority":
                        next_portfolio *= (1 - F)
                        next_portfolio *= (1 + rate)
                        next_position = 1
                        switch = 1

                # conflict case, both want to stay
                elif (p1 < thresh_m1) & (p2 < thresh_m2):
                    if conflict_resolution == "sedentary":
                        next_position = 0
                        switch = 0

                    elif conflict_resolution == "p_max":
                        if p1 > p2:
                            next_position = 0
                            switch = 0
                        else:
                            next_portfolio *= (1 - F)
                            next_portfolio *= (1 + rate)
                            next_position = 1
                            switch = 1

                    elif conflict_resolution == "stable_priority":
                        next_position = 0
                        switch = 0

                    elif conflict_resolution == "crypto_priority":
                        next_portfolio *= (1 - F)
                        next_portfolio *= (1 + rate)
                        next_position = 1
                        switch = 1

                # m1 want to move to crypto, m2 want to stay on crypto
                elif (p1 >= thresh_m1) & (p2 < thresh_m2):
                    next_portfolio *= (1 - F)
                    next_portfolio *= (1 + rate)
                    next_position = 1
                    switch = 1
                    
            return next_portfolio, next_position, switch
        
        portfolio = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.float64)
        positions = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.uint64)
        switchs = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.uint64)
        portfolio[0, :], positions[0, :], switchs[0, :] = K, START_POSITION, 0
        
        for i, (proba1_m1, proba1_m2, next_rate) in enumerate(zip(probas_m1, probas_m2, next_rates)):
            sim = 0
            for thresh_stable_to_crypto in lst_thresh_stable_to_crypto:
                for thresh_crypto_to_stable in lst_thresh_crypto_to_stable:
                    portfolio[i+1, sim], positions[i+1, sim], switchs[i+1, sim] = S_cbm(
                        F,
                        proba1_m1,
                        proba1_m2,
                        next_rate,
                        portfolio[i, sim],
                        positions[i, sim],
                        thresh_stable_to_crypto,
                        thresh_crypto_to_stable,
                        conflict_resolution
                    )
                    sim += 1
        return portfolio, positions, switchs

    @staticmethod
    @njit
    def _apply_strategy_mas(K, F, START_POSITION, next_rates, n_sims, probas_m1, probas_m2, lst_thresh_stable_to_crypto, lst_thresh_crypto_to_stable,
                            lst_window_stable_to_crypto, lst_window_crypto_to_stable):
        """
        *************************************************************** MOVING AVERAGE SIGNAL STRATEGY EXPLANATION ***************************************************************
        - if at time t, we long crypto, listen predictions from model crypto to stable. Else, listen predictions from model stable to crypto.
        - in order to limit the transaction fees impact on performances, a move can happen only if the mean of the last probabilities from listened model is higher 
          than the current threshold. For computing the mean, the last 'window' probabilities are taken.
        - when a change occurs, an another change can't occur since 'window' periods. 
        ****************************************************************************************************************************************************************
        Parameters
        ----------
        K : int or float
            starting capital
        F : float
            transaction fees, not in %. Means for example 0.1% of transaction fees: F = 0.001.
        START_POSITION : int
            0 if starting on stable, 1 if starting on crypto
        next_rates : numpy.ndarray
            next interest rate
        n_sims : int
            number of simulations
        probas_m1 : numpy.ndarray
            predictions from model crypto_to_stable
        probas_m2 : numpy.ndarray
            predictions from model stable_to_crypto
        lst_thresh_stable_to_crypto : numba.typed.List
            numba list of thresholds for stable_to_crypto, regarding dict of parameters
        lst_thresh_crypto_to_stable : numba.typed.List
            numba list of thresholds for crypto_to_stable, regarding dict of parameters
        lst_window_stable_to_crypto : numba.typed.List
            numba list of window for stable_to_crypto, regarding dict of parameters
        lst_window_crypto_to_stable : numba.typed.List
            numba list of window for crypto_to_stable, regarding dict of parameters
        """
        def S_mas(F, p1, p2, next_rate, portfolio_i, position_i, dq_p1, dq_p2, thresh_m1, thresh_m2, window_stable_to_crypto, window_crypto_to_stable):
            # NB : in this way, njit decorator is also applied to this function.
            #      numba 0.53.0 doesn't support python object of collections.deque()
            next_portfolio = portfolio_i
            dq_p1.append(p1)
            dq_p2.append(p2)

            if len(dq_p1) > window_stable_to_crypto:
                dq_p1.pop(0)

            if len(dq_p2) > window_crypto_to_stable:
                dq_p2.pop(0)

            if position_i:
                if len(dq_p2) < window_crypto_to_stable:
                    metric_2 = 0
                else:
                    metric_2 = np.mean(np.asarray(dq_p2))

                if metric_2 > thresh_m2:
                    next_portfolio *= (1 - F)
                    next_position = 0
                    switch = 1
                    dq_p1.clear()
                    dq_p2.clear()
                else:
                    next_portfolio *= (1 + next_rate)
                    next_position = 1
                    switch = 0
            else:
                if len(dq_p1) < window_stable_to_crypto:
                    metric_1 = 0
                else:
                    metric_1 = np.mean(np.asarray(dq_p1))

                if metric_1 > thresh_m1:
                    next_portfolio *= (1 - F)
                    next_portfolio *= (1 + next_rate)
                    next_position = 1
                    switch = 1
                    dq_p1.clear()
                    dq_p2.clear()
                else:
                    next_position = 0
                    switch = 0

            return next_portfolio, next_position, switch, dq_p1, dq_p2
        
        portfolio = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.float64)
        positions = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.uint64)
        switchs = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.uint64)
        portfolio[0, :], positions[0, :], switchs[0, :] = K, START_POSITION, 0
        
        lst_dq_p1, lst_dq_p2 = typed.List(),  typed.List()
        for _ in range(n_sims):
            lst_dq_p1.append(typed.List.empty_list(types.float64))
            lst_dq_p2.append(typed.List.empty_list(types.float64))
    
        for i, (proba1_m1, proba1_m2, next_rate) in enumerate(zip(probas_m1, probas_m2, next_rates)):
            sim = 0
            for thresh_stable_to_crypto in lst_thresh_stable_to_crypto:
                for thresh_crypto_to_stable in lst_thresh_crypto_to_stable:
                    for window_stable_to_crypto in lst_window_stable_to_crypto:
                        for window_crypto_to_stable in lst_window_crypto_to_stable:
                            portfolio[i+1, sim], positions[i+1, sim], switchs[i+1, sim], lst_dq_p1[sim], lst_dq_p2[sim] = S_mas(
                                F,
                                proba1_m1,
                                proba1_m2,
                                next_rate,
                                portfolio[i, sim],
                                positions[i, sim],
                                lst_dq_p1[sim],
                                lst_dq_p2[sim],
                                thresh_stable_to_crypto,
                                thresh_crypto_to_stable,
                                window_stable_to_crypto,
                                window_crypto_to_stable
                            )
                            sim += 1
        return portfolio, positions, switchs
    
    @staticmethod
    @njit
    def _apply_strategy_ols(K, F, START_POSITION, next_rates, n_sims, probas_m1, probas_m2, lst_thresh_stable_to_crypto, lst_thresh_crypto_to_stable,
                            lst_limit_stable_to_crypto, lst_limit_crypto_to_stable):
        """
        *************************************************************** OVER LIMIT SIGNAL STRATEGY EXPLANATION ***************************************************************
        - if at time t, we long on crypto, listen predictions from model crypto to stable. Else listen predictions from model stable to crypto.
        - in order to limit the transaction fees impact on performances, a move can happen only if the counter is over.
        - when a change occurs, an another change can't occur since the current element of lst_limit_stable_to_crypto or lst_limit_crypto_to_stable periods. 
        ****************************************************************************************************************************************************************
        Parameters
        ----------
        K : int or float
            starting capital
        F : float
            transaction fees, not in %. Means for example 0.1% of transaction fees: F = 0.001.
        START_POSITION : int
            0 if starting on stable, 1 if starting on crypto
        next_rates : numpy.ndarray
            next interest rate
        n_sims : int
            number of simulations
        probas_m1 : numpy.ndarray
            predictions from model crypto_to_stable
        probas_m2 : numpy.ndarray
            predictions from model stable_to_crypto
        lst_thresh_stable_to_crypto : numba.typed.List
            numba list of thresholds for stable_to_crypto, regarding dict of parameters
        lst_thresh_crypto_to_stable : numba.typed.List
            numba list of thresholds for crypto_to_stable, regarding dict of parameters
        lst_cons_stable_to_crypto : numba.typed.List
            numba list of consecutive_parameter for stable_to_crypto, regarding dict of parameters
        lst_cons_crypto_to_stable : numba.typed.List
            numba list of consecutive_parameter for crypto_to_stable, regarding dict of parameters
        """
        def S_ols(F, p1, p2, next_rate, portfolio_i, position_i, thresh_m1, thresh_m2, limit_1, limit_2, counter):
            # NB : in this way, njit decorator is also applied to this function.
            next_portfolio = portfolio_i
            if p1 > thresh_m1:
                counter[0] += 1
            if p2 > thresh_m2:
                counter[1] += 1
                            
            if position_i:
                if counter[1] > limit_2:
                    next_portfolio *= (1 - F)
                    next_position = 0
                    switch = 1
                    counter[1] = 0  # reset
                else:
                    next_portfolio *= (1 + next_rate)
                    next_position = 1
                    switch = 0
                    
            else:
                if counter[0] > limit_1:
                    next_portfolio *= (1 - F)
                    next_portfolio *= (1 + next_rate)
                    next_position = 1
                    switch = 1
                    counter[0] = 0  # reset
                else:
                    next_position = 0
                    switch = 0
            return next_portfolio, next_position, switch, counter
        
        portfolio = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.float64)
        positions = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.uint64)
        switchs = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.uint64)
        portfolio[0, :], positions[0, :], switchs[0, :] = K, START_POSITION, 0

        counters = np.zeros((2, n_sims), dtype=np.uint64)
    
        for i, (proba1_m1, proba1_m2, next_rate) in enumerate(zip(probas_m1, probas_m2, next_rates)):
            sim = 0
            for thresh_stable_to_crypto in lst_thresh_stable_to_crypto:
                for thresh_crypto_to_stable in lst_thresh_crypto_to_stable:
                    for limit_stable_to_crypto in lst_limit_stable_to_crypto:
                        for limit_crypto_to_stable in lst_limit_crypto_to_stable:
                            portfolio[i+1, sim], positions[i+1, sim], switchs[i+1, sim], counters[:, sim] = S_ols(
                                F,
                                proba1_m1,
                                proba1_m2,
                                next_rate,
                                portfolio[i, sim],
                                positions[i, sim],
                                thresh_stable_to_crypto,
                                thresh_crypto_to_stable,
                                limit_stable_to_crypto,
                                limit_crypto_to_stable,
                                counters[:, sim]
                            )
                            sim += 1
        return portfolio, positions, switchs
    
    @staticmethod
    @njit
    def _apply_strategy_css(K, F, START_POSITION, next_rates, n_sims, probas_m1, probas_m2, lst_thresh_stable_to_crypto, lst_thresh_crypto_to_stable,
                            lst_cons_stable_to_crypto, lst_cons_crypto_to_stable):
        """
        *************************************************************** CUMULATIVE STREAK SIGNAL STRATEGY EXPLANATION ***************************************************************
        - if at time t, we long crypto, listen predictions from model crypto to stable. Else, listen predictions from model stable to crypto.
        - in order to limit the transaction fees impact on performances, a move can happen only if last probabilities from listened model are all higher than the current threshold.
          The number of probability taken is given by lst_cons_stable_to_crypto or lst_cons_crypto_to_stable.
        - when a change occurs, an another change can't occur since the current element of lst_cons_stable_to_crypto or lst_cons_crypto_to_stable periods. 
        ****************************************************************************************************************************************************************
        Parameters
        ----------
        K : int or float
            starting capital
        F : float
            transaction fees, not in %. Means for example 0.1% of transaction fees: F = 0.001.
        START_POSITION : int
            0 if starting on stable, 1 if starting on crypto
        next_rates : numpy.ndarray
            next interest rate
        n_sims : int
            number of simulations
        probas_m1 : numpy.ndarray
            predictions from model crypto_to_stable
        probas_m2 : numpy.ndarray
            predictions from model stable_to_crypto
        lst_thresh_stable_to_crypto : numba.typed.List
            numba list of thresholds for stable_to_crypto, regarding dict of parameters
        lst_thresh_crypto_to_stable : numba.typed.List
            numba list of thresholds for crypto_to_stable, regarding dict of parameters
        lst_cons_stable_to_crypto : numba.typed.List
            numba list of consecutive_parameter for stable_to_crypto, regarding dict of parameters
        lst_cons_crypto_to_stable : numba.typed.List
            numba list of consecutive_parameter for crypto_to_stable, regarding dict of parameters
        """
        def S_css(F, p1, p2, next_rate, portfolio_i, position_i, thresh_m1, thresh_m2, cons_1, cons_2, counter):
            # NB : in this way, njit decorator is also applied to this function.
            next_portfolio = portfolio_i
            if p1 > thresh_m1:
                counter[0] += 1
            else:
                counter[0] = 0  # reset
            if p2 > thresh_m2:
                counter[1] += 1
            else:
                counter[1] = 0  # reset
                            
            if position_i:
                if counter[1] > cons_2:
                    next_portfolio *= (1 - F)
                    next_position = 0
                    switch = 1
                else:
                    next_portfolio *= (1 + next_rate)
                    next_position = 1
                    switch = 0
                    
            else:
                if counter[0] > cons_1:
                    next_portfolio *= (1 - F)
                    next_portfolio *= (1 + next_rate)
                    next_position = 1
                    switch = 1
                else:
                    next_position = 0
                    switch = 0
            return next_portfolio, next_position, switch, counter
        
        portfolio = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.float64)
        positions = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.uint64)
        switchs = np.empty(shape=(len(probas_m1)+1, n_sims), dtype=np.uint64)
        portfolio[0, :], positions[0, :], switchs[0, :] = K, START_POSITION, 0
        counters = np.zeros((2, n_sims), dtype=np.uint64)
    
        for i, (proba1_m1, proba1_m2, next_rate) in enumerate(zip(probas_m1, probas_m2, next_rates)):
            sim = 0
            for thresh_stable_to_crypto in lst_thresh_stable_to_crypto:
                for thresh_crypto_to_stable in lst_thresh_crypto_to_stable:
                    for cons_stable_to_crypto in lst_cons_stable_to_crypto:
                        for cons_crypto_to_stable in lst_cons_crypto_to_stable:
                            portfolio[i+1, sim], positions[i+1, sim], switchs[i+1, sim], counters[:, sim] = S_css(
                                F,
                                proba1_m1,
                                proba1_m2,
                                next_rate,
                                portfolio[i, sim],
                                positions[i, sim],
                                thresh_stable_to_crypto,
                                thresh_crypto_to_stable,
                                cons_stable_to_crypto,
                                cons_crypto_to_stable,
                                counters[:, sim]
                            )
                            sim += 1
        return portfolio, positions, switchs

    def __call__(self, d_params, pred1_model_stable_to_crypto, pred1_model_crypto_to_stable):
        """
        make simulations regarding the dict of parameters and predictions (proba) given.
        Parameters
        ----------
        d_params : dict
            dict of parameters, one strategy max is allowed
        pred1_model_stable_to_crypto : pandas.Series
            prediction from model stable to crypto
        pred1_model_crypto_stable : pandas.Series
            prediction from model crypto to stable
        Returns
        -------
        numpy.ndarray[n_preds, n_sims], numpy.ndarray[n_preds, n_sims], numpy.ndarray[n_preds, n_sims], list
            all the simulation done (one simulation per column)
            all the position from each time from each simulation
            all the move from each time from each simulation
            ordered list of parameters
        """
        assert len(d_params) == 1, "the dictionary of parameters must have the following structure :\n\
            {\n\
                <strategy_name>: {\n\
                    <parameter_0>: ...,\n\
                    <parameter_1>: ...,\n\
                    ...,\n\
                    <parameter_n>: ...\n\
                }\n\
            }"
        strategy = self._check_strategy(d_params)
        n_sims = __class__._get_n_sims(d_params[strategy])
        next_rates = self.rates.shift(-1).fillna(0).values  # next return, last value filled  
        
        if strategy == 'traditional':
            numba_lst_thresh_stable_to_crypto, numba_lst_thresh_crypto_to_stable = __class__._python_params_to_numba_traditional(d_params[strategy])
            lst_parameters_combination = self._get_lst_params_strategy_traditional(d_params['traditional']) 
            portfolio, positions, switchs = __class__._apply_strategy_traditional(
                K=self.K,
                F=self.F,
                START_POSITION=self.START_POSITION,
                next_rates=next_rates,
                n_sims=n_sims,
                probas_m1=pred1_model_stable_to_crypto.values,
                probas_m2=pred1_model_crypto_to_stable.values,
                lst_thresh_stable_to_crypto=numba_lst_thresh_stable_to_crypto,
                lst_thresh_crypto_to_stable=numba_lst_thresh_crypto_to_stable
            )
            
        elif strategy == 'combine_both_models':
            numba_lst_thresh_stable_to_crypto, numba_lst_thresh_crypto_to_stable, conflict_resolution = __class__._python_params_to_numba_cbm(d_params[strategy])
            lst_parameters_combination = self._get_lst_params_strategy_cbm(d_params['combine_both_models']) 
            portfolio, positions, switchs = __class__._apply_strategy_cbm(
                K=self.K,
                F=self.F,
                START_POSITION=self.START_POSITION,
                next_rates=next_rates,
                n_sims=n_sims,
                probas_m1=pred1_model_stable_to_crypto.values,
                probas_m2=pred1_model_crypto_to_stable.values,
                lst_thresh_stable_to_crypto=numba_lst_thresh_stable_to_crypto,
                lst_thresh_crypto_to_stable=numba_lst_thresh_crypto_to_stable,
                conflict_resolution=conflict_resolution
            )
    
        elif strategy == 'moving_average_signal':
            numba_lst_thresh_stable_to_crypto, numba_lst_thresh_crypto_to_stable, numba_lst_window_stable_to_crypto, numba_lst_window_crypto_to_stable = \
                                                                                                        __class__._python_params_to_numba_mas(d_params[strategy])
            lst_parameters_combination = self._get_lst_params_strategy_mas(d_params['moving_average_signal'])                                                                                            
            portfolio, positions, switchs = __class__._apply_strategy_mas(
                K=self.K,
                F=self.F,
                START_POSITION=self.START_POSITION,
                next_rates=next_rates,
                n_sims=n_sims,
                probas_m1=pred1_model_stable_to_crypto.values,
                probas_m2=pred1_model_crypto_to_stable.values,
                lst_thresh_stable_to_crypto=numba_lst_thresh_stable_to_crypto,
                lst_thresh_crypto_to_stable=numba_lst_thresh_crypto_to_stable,
                lst_window_stable_to_crypto=numba_lst_window_stable_to_crypto,
                lst_window_crypto_to_stable=numba_lst_window_crypto_to_stable
            )
            
        elif strategy == 'over_limit_signal':
            numba_lst_thresh_stable_to_crypto, numba_lst_thresh_crypto_to_stable, numba_lst_limit_stable_to_crypto, numba_lst_limit_crypto_to_stable = \
                                                                                                        __class__._python_params_to_numba_ols(d_params[strategy])
            lst_parameters_combination = self._get_lst_params_strategy_ols(d_params['over_limit_signal'])
            portfolio, positions, switchs = __class__._apply_strategy_ols(
                K=self.K,
                F=self.F,
                START_POSITION=self.START_POSITION,
                next_rates=next_rates,
                n_sims=n_sims,
                probas_m1=pred1_model_stable_to_crypto.values,
                probas_m2=pred1_model_crypto_to_stable.values,
                lst_thresh_stable_to_crypto=numba_lst_thresh_stable_to_crypto,
                lst_thresh_crypto_to_stable=numba_lst_thresh_crypto_to_stable,
                lst_limit_stable_to_crypto=numba_lst_limit_stable_to_crypto,
                lst_limit_crypto_to_stable=numba_lst_limit_crypto_to_stable
            )
            
        elif strategy == 'consecutive_streak_signal':
            numba_lst_thresh_stable_to_crypto, numba_lst_thresh_crypto_to_stable, numba_lst_cons_stable_to_crypto, numba_lst_cons_crypto_to_stable = \
                                                                                                        __class__._python_params_to_numba_css(d_params[strategy])
            lst_parameters_combination = self._get_lst_params_strategy_css(d_params['consecutive_streak_signal'])                                                                                        
            portfolio, positions, switchs = __class__._apply_strategy_css(
                K=self.K,
                F=self.F,
                START_POSITION=self.START_POSITION,
                next_rates=next_rates,
                n_sims=n_sims,
                probas_m1=pred1_model_stable_to_crypto.values,
                probas_m2=pred1_model_crypto_to_stable.values,
                lst_thresh_stable_to_crypto=numba_lst_thresh_stable_to_crypto,
                lst_thresh_crypto_to_stable=numba_lst_thresh_crypto_to_stable,
                lst_cons_stable_to_crypto=numba_lst_cons_stable_to_crypto,
                lst_cons_crypto_to_stable=numba_lst_cons_crypto_to_stable
            )

        return portfolio, positions, switchs, lst_parameters_combination
