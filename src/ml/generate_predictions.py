#!/usr/bin/python3.8


import os

import numpy as np
import pandas as pd
from pyhocon import ConfigFactory
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from utils.conf_logger import configure_logger
from cl_training import WindowTrainingSpec



def check_exists_or_create(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return None


if __name__ == "__main__":
    # settings
    conf = ConfigFactory.parse_file("conf/predict.conf")
    logger = configure_logger(filename=os.path.basename(__file__))
    
    model_stable_to_crypto = LogisticRegression(class_weight="balanced", max_iter=500)
    model_crypto_to_stable = LogisticRegression(class_weight="balanced", max_iter=500)
    COL_X = list(set(conf['col_x_crypto_to_stable'] + conf['col_x_stable_to_crypto']))

    # run
    logger.info("load data ...")
    df_market = pd.read_csv(conf['path_data'], index_col=0, parse_dates=['open_time', 'close_time'])
    df_market.index = df_market.close_time

    X = df_market.loc[conf['start']:conf['end'], COL_X]
    y_stable_to_crypto = df_market.loc[conf['start']:conf['end'], 'actions_stable_to_crypto']
    y_crypto_to_stable = df_market.loc[conf['start']:conf['end'], 'actions_crypto_to_stable']
    w_stable_to_crypto = df_market.loc[conf['start']:conf['end'], 'weights_stable_to_crypto']
    w_crypto_to_stable = df_market.loc[conf['start']:conf['end'], 'weights_crypto_to_stable']
    index_knowledge_stable_to_crypto = df_market.loc[conf['start']:conf['end'], 'index_knowledge_stable_to_crypto']
    index_knowledge_crypto_to_stable = df_market.loc[conf['start']:conf['end'], 'index_knowledge_crypto_to_stable']
    logger.info("... done")
    
    window_training_spec = WindowTrainingSpec(size_window=conf['size_window'], refit_fqz=conf['refit_fqz'])
    logger.info("predict with model stable to crypto ...")
    preds_stable_to_crypto, dic_imp_stable_to_crypto = window_training_spec.fit_predict(
        model=model_stable_to_crypto,
        X=X[conf['col_x_stable_to_crypto']],
        y=y_stable_to_crypto.values,
        index_knowledge=index_knowledge_stable_to_crypto.values,
        sample_weight=w_stable_to_crypto.values,
        fit_or_refit="fit",
        predict_proba=True,
        accumulate_window=False,
        get_feat_imp=True
    )
    logger.info("... done")
    
    logger.info("predict with model crypto to stable ...")
    preds_crypto_to_stable, dic_imp_crypto_to_stable = window_training_spec.fit_predict(
        model=model_crypto_to_stable,
        X=X[conf['col_x_crypto_to_stable']],
        y=y_crypto_to_stable.values,
        index_knowledge=index_knowledge_crypto_to_stable.values,
        sample_weight=w_crypto_to_stable.values,
        fit_or_refit="fit",
        predict_proba=True,
        accumulate_window=False,
        get_feat_imp=True
    )
    logger.info("... done")
    
    logger.info("saving ...")
    check_exists_or_create(conf['save_path'])
    path_preds_stable_to_crypto = os.path.join(
        conf['save_path'],
        f"preds_stable_to_crypto_{conf['market']}_{conf['frequency']}_{conf['fees']}_{conf['start']}_to_{conf['end']}_{conf['size_window']}_{conf['refit_fqz']}.csv"
        )
    path_preds_crypto_to_stable = os.path.join(
        conf['save_path'],
        f"preds_crypto_to_stable_{conf['market']}_{conf['frequency']}_{conf['fees']}_{conf['start']}_to_{conf['end']}_{conf['size_window']}_{conf['refit_fqz']}.csv"
        )
    pd.DataFrame(preds_stable_to_crypto).to_csv(path_preds_stable_to_crypto)
    pd.DataFrame(preds_crypto_to_stable).to_csv(path_preds_crypto_to_stable)
    logger.info("... done")
