#!/usr/bin/python3.8


import os
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhocon import ConfigFactory

from utils.conf_logger import configure_logger
from cl_strategy_simulator import StrategySimulator
from cl_strategy_selector import StrategySelector
from cl_strategy_evaluator import StrategyEvaluator


if __name__ == "__main__":
    # settings
    conf = ConfigFactory.parse_file("conf/strategy.conf")
    logger = configure_logger(filename=os.path.basename(__file__))
    DIC_PARAMS = {
        "traditional": {
            "thresh_m1": np.arange(.1, .9, .01),
            "thresh_m2": np.arange(.1, .9, .01),
        },
        # "combine_both_models": {
        #     "thresh_m1": np.arange(.01, .95, .01),
        #     "thresh_m2": np.arange(.01, .95, .01),
        #     "conflict_resolution": "sedentary"
        # },
        # "moving_average_signal": {
        #     "thresh_m1": np.arange(.2, .9, .01),
        #     "thresh_m2": np.arange(.2, .9, .01),
        #     "window_stable_to_crypto": 20,
        #     "window_crypto_to_stable": 20
        # },
        # "over_limit_signal": {
        #     "thresh_m1": np.arange(.1, .9, .01),
        #     "thresh_m2": np.arange(.1, .9, .01),
        #     "limit_stable_to_crypto": 30,
        #     "limit_crypto_to_stable": 30
        # },
        # "consecutive_streak_signal": {
        #     "thresh_m1": np.arange(.1, .9, .01),
        #     "thresh_m2": np.arange(.1, .9, .01),
        #     "n_cons_stable_to_crypto": 5,
        #     "n_cons_crypto_to_stable": 5
        # }
    }
    path_data = conf["path_preprocessed_data"]
    path_preds_stable_to_crypto = conf["path_pred_stable_to_crypto"]
    path_preds_crypto_to_stable = conf["path_pred_crypto_to_stable"]

    # load
    logger.info("load data ...")
    data = pd.read_csv(path_data)
    data.set_index('close_time', drop=False, inplace=True)
    preds_stable_to_crypto = pd.read_csv(path_preds_stable_to_crypto, index_col=0)
    preds_crypto_to_stable = pd.read_csv(path_preds_crypto_to_stable, index_col=0)
    rates = data.loc[preds_crypto_to_stable.index, f"rdt_crypto_{conf['frequency']}"]
    close = data.loc[preds_crypto_to_stable.index, "close"]
    logger.info("... done.")

    # run
    logger.info("simulate ...")
    start = time.time()
    stock_norm = StrategySelector.normalize_stock_market(
        stock=close.values,
        K=conf['K']
    )
    simulator = StrategySimulator(
        K=conf['K'],
        F=conf['fees'],
        rates=rates
    )
    portfolio_simulations, positions, switchs, params_combination = simulator(
        d_params=DIC_PARAMS,
        pred1_model_stable_to_crypto=preds_stable_to_crypto['0'],
        pred1_model_crypto_to_stable=preds_crypto_to_stable['0']
    )
    end = time.time()
    logger.info(f"... {len(params_combination)} simulations done in : {round(end - start, 3)}s")
    
    # display?
    if conf['display_simulation']:
        logger.info("plot random simulation and normalized quoted asset ...")
        lst_random_sims = [random.randint(0, portfolio_simulations.shape[1]) for _ in range(conf['n_random'])]
        plt.figure(figsize=(20, 5))
        plt.plot(portfolio_simulations[:, lst_random_sims], label='random simulation', linestyle='dashed')
        plt.plot(stock_norm, label='stock norm', c='black')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend(loc='upper left')
        plt.show()

    logger.info("find best simulation ...")
    dic_coefs = StrategySelector.get_dic_coefs(
        stock=close.values,
        K=conf['K'],
        portfolio=portfolio_simulations,
        filter_losing=conf['filter_losing']
    )
    best_sims, best_params = StrategySelector.best_n(
        n=conf["top_n"],
        stock=close.values,
        K=conf['K'],
        portfolio=portfolio_simulations,
        lst_parameters_simulation=params_combination,
        filter_losing=conf['filter_losing']
    )
    worst_sims, worst_params = StrategySelector.worst_n(
        n=conf['top_n'],
        stock=close.values,
        K=conf['K'],
        portfolio=portfolio_simulations,
        lst_parameters_simulation=params_combination,
        filter_losing=conf['filter_losing']
    )
    print(f'\ntop {conf["top_n"]} simulations : {best_sims} best parameters are : {best_params}\n')
    print(f'worst {conf["worst_n"]} simulations : {worst_sims} worst parameters are : {worst_params}\n')
    
    # display?
    if conf['display_simulation']:
        logger.info("plot best/worst simulation and normalized quoted asset ...")
        plt.figure(figsize=(20,5))
        plt.plot(portfolio_simulations[:, best_sims[0]], label='best_simulation', linestyle='dashed', c='green')
        plt.plot(portfolio_simulations[:, worst_sims[0]], label=f'worst_simulation (according param filter_losing)', linestyle='dashed', c='red')
        plt.plot(stock_norm, label='stock norm', c='black')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend(loc='upper left')
        plt.show()
    
    resume = pd.concat(
            [
                pd.Series(stock_norm, index=preds_crypto_to_stable.index),
                pd.Series(portfolio_simulations[:-1, best_sims[0]], index=preds_crypto_to_stable.index),
                pd.Series(positions[:-1, best_sims[0]], index=preds_crypto_to_stable.index),
                pd.Series(switchs[:-1, best_sims[0]], index=preds_crypto_to_stable.index),
            ],
            1
    )
    resume.columns = ['close_norm', 'portfolio', 'position', 'switch']
    data = pd.concat([resume, data.loc[preds_crypto_to_stable.index, :]], 1)
    data.reset_index(drop=True, inplace=True)
    
    logger.info("evaluate best strategy ...")
    params_report = conf['params_report']
    params_report["best_params_simulation"] = best_params[0]
    strategy_eval = StrategyEvaluator(
        data=data,
        params_report=conf['params_report']
    )
    strategy_eval.generate_report("data/evaluation")
    logger.info("done.")
