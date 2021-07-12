#!/usr/bin/python3.8


import numpy as np
from sklearn.linear_model import LinearRegression


class StrategySelector:
    
    """
        Simple way for trying to select the best portfolio simulation and getting the best parameters.
        Can be also use for hypertuning parameters of simulations. 
        METHOD : 
            Foreach simulated portfolio evolution:
                . compute difference between simulated portfolio evolution and stock market.
                . fit a linear regression on this difference without intercept (i.e. y = a*x).
                . save coef 'a'.
            best simulations are higher coef 'a'. (resp worst)
    """
    
    @staticmethod
    def _filter_on_x(x, portfolio, lst_sims_number):
        """
        keep only the portfolio simulations where the terminal value is greater than x.
        """
        lst_allowed = []
        for sim_i in lst_sims_number:
            terminal_value_sim_i = portfolio[-1, sim_i]
            if terminal_value_sim_i > x:
                lst_allowed.append(sim_i)
        return lst_allowed
    
    @staticmethod
    def _check_parameter_n(n, dic):
        """
        raise error if n > len(dic).
        """
        if n > len(dic):
            raise ValueError(f"max allowed size for parameter 'n' is {len(dic)}.")
        return None
    
    @staticmethod
    def normalize_stock_market(stock, K):
        """
        normalize stock in order to be able to compare crypto evolution with portfolio evolution.
        """
        coeff = stock[0] / K
        stock_norm = stock / coeff
        return stock_norm
    
    @staticmethod
    def get_dic_coefs(stock, K, portfolio, filter_losing=True):
        """
        cf. METHOD.
        Parameters
        ----------
        stock : np.ndarray
            open, close ...
        K : int or float
            starting capital
        portfolio : np.ndarray[len(stock)+1, # simulations]
            simulated portfolio evolution
        filter_losing : bool, optional
            before apply METHOD, filter all losing simulations (i.e. < K & < stock[-1]), by default True
        Returns
        -------
        dict
            keys: simulation number
            values: coeff a
        """
        stock_norm = __class__.normalize_stock_market(stock, K)
        
        if filter_losing:
            lst_allowed_sim = __class__._filter_on_x(
                x=max(K, stock_norm[-1]),
                portfolio=portfolio,
                lst_sims_number=np.arange(portfolio.shape[1]).tolist()
            )
        else:
            lst_allowed_sim = np.arange(portfolio.shape[1]).tolist()

        dic_coefs = {}
        for allowed_sim in lst_allowed_sim:
            diff = portfolio[1:, allowed_sim] - stock_norm
            x = np.arange(len(diff))
            clf = LinearRegression(fit_intercept=False)
            clf_fitted = clf.fit(x.reshape(-1,1), diff.reshape(-1,1))
            dic_coefs[str(allowed_sim)] = clf_fitted.coef_.ravel()[0]
        
        # sort dict
        dic_coefs = dict(sorted(dic_coefs.items(), key=lambda x: x[1], reverse=True))
        return dic_coefs
    
    @staticmethod
    def best_n(n, stock, K, portfolio, lst_parameters_simulation, filter_losing=True):
        """
        get best n simulations regarding METHOD.
        Parameters
        ----------
        n : int
            get best 'n' sim.
        stock : np.ndarray
            open, close ...
        K : int or float
            starting capital
        portfolio : np.ndarray[len(stock)+1, # simulations]
            simulated portfolio evolution
        lst_parameters_simulation : list
            params used for each simulations
        filter_losing : bool, optional
            before apply METHOD, filter all losing simulations (i.e. < K & < stock[-1]), by default True
        Returns
        -------
        list, list
            best simulations number, best params.
        """
        dic_coefs = __class__.get_dic_coefs(stock, K, portfolio, filter_losing)
        __class__._check_parameter_n(n, dic_coefs)
        
        lst_best = []
        for i, sim in enumerate(dic_coefs.keys()):
            if i < n:
                lst_best.append(int(sim))
            else:
                break
        return lst_best, np.array(lst_parameters_simulation)[lst_best].tolist()
    
    @staticmethod
    def worst_n(n, stock, K, portfolio, lst_parameters_simulation, filter_losing=False):
        """
        get worst n simulations regarding METHOD.
        Parameters
        ----------
        n : int
            get worst 'n' sim.
        stock : np.ndarray
            open, close ...
        K : int or float
            starting capital
        portfolio : np.ndarray[len(stock)+1, # simulations]
            simulated portfolio evolution
        lst_parameters_simulation : list
            params used for each simulations
        filter_losing : bool, optional
            before apply METHOD, filter all losing simulations (i.e. < K & < stock[-1]), by default True
        Returns
        -------
        list, list
            worst simulations number, worst params.
        """
        dic_coefs = __class__.get_dic_coefs(stock, K, portfolio, filter_losing)
        dic_coefs = dict(sorted(dic_coefs.items(), key=lambda x: x[1], reverse=False))
        __class__._check_parameter_n(n, dic_coefs)

        lst_worst = []
        for i, sim in enumerate(dic_coefs.keys()):
            if i < n:
                lst_worst.append(int(sim))
            else:
                break
        return lst_worst, np.array(lst_parameters_simulation)[lst_worst].tolist()
