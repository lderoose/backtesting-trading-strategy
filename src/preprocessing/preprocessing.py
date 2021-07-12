#!/usr/bin/python3.8


import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from pyhocon import ConfigFactory

from cl_labelling import Labelling
from cl_features_engineering import TimeSeriesFeaturesEngineering
from utils.conf_logger import configure_logger
from src.database.cl_binance_database import BinanceDatabase

# TODO : check if data is available in database regarding conf.start and conf.end


def strDatetime_to_strTimestamp(d, format='%Y-%m-%d_%H:%M:%S', coeff_unit=1000):
    ts = datetime.strptime(d, format)
    return str(coeff_unit*int(ts.timestamp()))


def to_dataframe(output_request, lst_columns):
    # don't take market, frequency and open_time_human columns
    return pd.DataFrame(np.array(output_request)[:, 2:-1], columns=lst_columns)


def _count_nan(df):
    return df.isna().sum().cumsum()[-1]


def fill_nan(df, lst_columns):
    df = df.sort_values(by="open_time")
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms',  utc=True).dt.tz_localize(None)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True).dt.tz_localize(None)
    
    lst_cols_without_time = lst_columns.copy()
    lst_cols_without_time.remove("open_time")
    lst_cols_without_time.remove("close_time")
    
    # reindex
    df.index = df.open_time
    
    # find beginning of df in Binance (assume no remove of market)
    start_market_open = min(df.open_time)
    end_market_open = max(df.open_time)
    start_market_close = min(df.close_time)
    end_market_close = max(df.close_time)
    
    # join on full interval of frequency
    # for adding new frequency, see: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    if conf["frequency"] == "1m":
        date_range_open = pd.date_range(start_market_open, end_market_open, freq='1min')
        date_range_close = pd.date_range(start_market_close, end_market_close, freq='1min')
    elif conf["frequency"] == "1h":
        date_range_open = pd.date_range(start_market_open, end_market_open, freq='H')
        date_range_close = pd.date_range(start_market_close, end_market_close, freq='H')
    elif conf["frequency"] == "1d":
        date_range_open = pd.date_range(start_market_open, end_market_open, freq='D')
        date_range_close = pd.date_range(start_market_close, end_market_close, freq='D')
    else:
        raise ValueError("define date range for this frequency !")
    
    df_tmp = pd.DataFrame(columns=['to_del'], index=date_range_open, dtype=object)
    df = df_tmp.join(df, how="left")
    df.drop('to_del', axis=1, inplace=True)
    df["open_time"] = date_range_open
    df["close_time"] = date_range_close
    
    # fillnan
    n_missing_before = _count_nan(df)
    for col in lst_cols_without_time:
        df[col].fillna(method="ffill", inplace=True)
    n_missing_after = _count_nan(df) 
    
    if n_missing_after == 0:
        n_rows_added = n_missing_before / df.shape[0]
        if n_rows_added > 10:
            raise ValueError(f"too NaN in data ({n_rows_added} %), please check.")
        else:
            return df
    else:
        raise ValueError(f"remaining {n_missing_after} NaN, check function preprocessing.fill_nan.")
    
    
def clean_types(df):
    lst_float_cols = [
        "open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    for col in lst_float_cols:
        df[col] = df[col].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)
    return df
    

def check_exists_or_create(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return None


def features_engineering_1m(data, idx, stock):
    """
    add new features in data if frequency = 1m.
    """
    data.loc[idx, "rdt_crypto_1m"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*1)       
    data.loc[idx, "rdt_crypto_5m"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*5)
    data.loc[idx, "rdt_crypto_15m"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*15)
    data.loc[idx, "rdt_crypto_30m"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*30)
    data.loc[idx, "rdt_crypto_1h"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*60)
    data.loc[idx, "rdt_crypto_2h"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*60*2)
    data.loc[idx, "rdt_crypto_4h"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*60*4)
    data.loc[idx, "rdt_crypto_6h"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*60*6)
    data.loc[idx, "rdt_crypto_12h"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*60*12)
    data.loc[idx, "rdt_crypto_1d"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*60*24)
    data.loc[idx, "rdt_crypto_3d"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*60*24*3)
    data.loc[idx, "rdt_crypto_1w"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*60*24*7)
    data.loc[idx, "rdt_crypto_1mo"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*60*24*30)
    
    # syntax : metric_timeseries_frequency_window
    data["vol_rdt_close_1m_15m"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1m.values,
        window=15,
        freq=1
    )
    data["vol_rdt_close_1m_1h"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1m.values,
        window=60,
        freq=1
    )
    data["vol_rdt_close_1m_1d"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1m.values,
        window=1440,
        freq=1
    )
    data["vol_rdt_close_1m_1w"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1m.values,
        window=7*1440,
        freq=1
    )

    data["vol_rdt_close_1h_12h"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1h.values,
        window=12,
        freq=60
    )
    data["vol_rdt_close_1h_1d"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1h.values,
        window=24,
        freq= 60
    )
    data["vol_rdt_close_1h_1w"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1h.values,
        window=7*24,
        freq=60
    )
    data["vol_rdt_close_1h_1month"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1h.values,
        window=7*24*4,
        freq=60
    )

    data["vol_rdt_close_1d_1w"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1d.values,
        window=7,
        freq=1440
    )
    data["vol_rdt_close_1d_1month"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1d.values,
        window=28,
        freq=1440
    )
    data["vol_rdt_close_1d_3month"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1d.values,
        window=3*28,
        freq=1440
    )

    # cumulative expected return
    data["cer_close_1m_15m"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1m.values,
        window=15,
        freq=1
    )
    data["cer_close_1m_1h"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1m.values,
        window=60,
        freq=1
    )
    data["cer_close_1m_1d"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1m.values,
        window=1440,
        freq=1
    )
    data["cer_close_1m_1w"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1m.values,
        window=7*1440,
        freq=1
    )

    data["cer_close_1h_12h"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1h.values,
        window=12,
        freq=60
    )
    data["cer_close_1h_1d"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1h.values,
        window=24,
        freq= 60
    )
    data["cer_close_1h_1w"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1h.values,
        window=7*24,
        freq=60
    )
    data["cer_close_1h_1month"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1h.values,
        window=7*24*4,
        freq=60
    )

    data["cer_close_1d_1w"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1d.values,
        window=7,
        freq=1440
    )
    data["cer_close_1d_1month"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1d.values,
        window=28,
        freq=1440
    )
    data["cer_close_1d_3month"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1d.values,
        window=3*28,
        freq=1440
    )

    # length streak
    data["positive_length_streak_rdt_close_1m"] = TimeSeriesFeaturesEngineering.compute_length_streak(
        timeseries=data.rdt_crypto_1m.values,
        freq=1,
        positive=1
    )
    data["negative_length_streak_rdt_close_1m"] = TimeSeriesFeaturesEngineering.compute_length_streak(
        timeseries=data.rdt_crypto_1m.values,
        freq=1,
        positive=0
    )
    data["positive_length_streak_rdt_close_1h"] = TimeSeriesFeaturesEngineering.compute_length_streak(
        timeseries=data.rdt_crypto_1h.values,
        freq=60,
        positive=1
    )
    data["negative_length_streak_rdt_close_1h"] = TimeSeriesFeaturesEngineering.compute_length_streak(
        timeseries=data.rdt_crypto_1h.values,
        freq=60,
        positive=0
    )
    data["positive_length_streak_rdt_close_1d"] = TimeSeriesFeaturesEngineering.compute_length_streak(
        timeseries=data.rdt_crypto_1d.values,
        freq=1440,
        positive=1
    )
    data["negative_length_streak_rdt_close_1d"] = TimeSeriesFeaturesEngineering.compute_length_streak(
        timeseries=data.rdt_crypto_1d.values,
        freq=1440,
        positive=0
    )
    return None


def features_engineering_1h(data, idx, stock):
    """
    add new features in data if frequency = 1h.
    """
    data.loc[idx, "rdt_crypto_1h"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1)
    data.loc[idx, "rdt_crypto_2h"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*2)
    data.loc[idx, "rdt_crypto_4h"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*4)
    data.loc[idx, "rdt_crypto_6h"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*6)
    data.loc[idx, "rdt_crypto_12h"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*12)
    data.loc[idx, "rdt_crypto_1d"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*24)
    data.loc[idx, "rdt_crypto_3d"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*24*3)
    data.loc[idx, "rdt_crypto_1w"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*24*7)
    data.loc[idx, "rdt_crypto_1mo"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*24*30)
    
    # # syntax : metric_timeseries_frequency_window
    data["vol_rdt_close_1h_12h"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1h.values,
        window=12,
        freq=1
    )
    data["vol_rdt_close_1h_1d"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1h.values,
        window=24,
        freq=1
    )
    data["vol_rdt_close_1h_3d"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1h.values,
        window=24*3,
        freq=1
    )
    data["vol_rdt_close_1h_1w"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1h.values,
        window=24*7,
        freq=1
    )

    data["vol_rdt_close_1h_1mo"] = TimeSeriesFeaturesEngineering.compute_volatility(
        timeseries=data.rdt_crypto_1h.values,
        window=24*7*4,
        freq=1
    )
    # cumulative expected return
    data["cer_close_1h_12h"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1h.values,
        window=12,
        freq=1
    )
    data["cer_close_1h_1d"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1h.values,
        window=24,
        freq=1
    )
    data["cer_close_1h_3d"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1h.values,
        window=24*3,
        freq=1
    )
    data["cer_close_1h_1w"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1h.values,
        window=24*7,
        freq=1
    )
    data["cer_close_1h_1mo"] = TimeSeriesFeaturesEngineering.compute_cumulative_expected_return(
        timeseries=data.rdt_crypto_1h.values,
        window=7*24*4,
        freq=1
    )

    # length streak
    data["positive_length_streak_rdt_close_1h"] = TimeSeriesFeaturesEngineering.compute_length_streak(
        timeseries=data.rdt_crypto_1h.values,
        freq=1,
        positive=1
    )
    data["negative_length_streak_rdt_close_1h"] = TimeSeriesFeaturesEngineering.compute_length_streak(
        timeseries=data.rdt_crypto_1h.values,
        freq=1,
        positive=0
    )
    data["positive_length_streak_rdt_close_12h"] = TimeSeriesFeaturesEngineering.compute_length_streak(
        timeseries=data.rdt_crypto_1h.values,
        freq=12,
        positive=1
    )
    data["negative_length_streak_rdt_close_12h"] = TimeSeriesFeaturesEngineering.compute_length_streak(
        timeseries=data.rdt_crypto_1h.values,
        freq=12,
        positive=0
    )
    data["positive_length_streak_rdt_close_1d"] = TimeSeriesFeaturesEngineering.compute_length_streak(
        timeseries=data.rdt_crypto_1h.values,
        freq=24,
        positive=1
    )
    data["negative_length_streak_rdt_close_1d"] = TimeSeriesFeaturesEngineering.compute_length_streak(
        timeseries=data.rdt_crypto_1h.values,
        freq=24,
        positive=0
    )
    return None

# example
def features_engineering_1d(data, idx, stock):
    data.loc[idx, "rdt_crypto_1d"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1)
    data.loc[idx, "rdt_crypto_3d"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*3)
    data.loc[idx, "rdt_crypto_1w"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*7)
    data.loc[idx, "rdt_crypto_1mo"] = TimeSeriesFeaturesEngineering.expected_return(data.loc[idx, stock], 1*30)
    # you can add new column here
    return None
    

if __name__ == "__main__":
    
    LST_COLUMNS = [
        "open_time",
        "open", 'high', 'low', 'close', "volume",
        "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    conf = ConfigFactory.parse_file("conf/preprocessing.conf")
    logger = configure_logger(filename=os.path.basename(__file__))
    
    # create connection to db
    db = BinanceDatabase(
        database=conf["database"],
        table=conf["table"],
        logger=logger
    )
    
    ts_start = strDatetime_to_strTimestamp(conf["start_date"])
    ts_end = strDatetime_to_strTimestamp(conf["end_date"])
    
    request = "SELECT * FROM '" + conf['table'] + "'\
                 WHERE open_time >= '" + ts_start + "' AND open_time <= '" + ts_end + "' \
                 AND market = '" + conf['market'] + "' \
                 AND frequency = '" + conf['frequency'] + "';\
              "
    # catch data
    start_time_read_db = time.time()
    logger.info("load data from SQL database ...")
    data = db.execute_sql_request(request)
    logger.info(f"... done in {round(time.time()-start_time_read_db, 3)}s.")
    
    # preprocessing 1: cleaning
    data = to_dataframe(output_request=data, lst_columns=LST_COLUMNS)
    start_time_nan = time.time()
    logger.info("filling NaN ...")
    data = fill_nan(data, LST_COLUMNS)
    logger.info(f"... done in {round(time.time()-start_time_nan, 3)}s.")
    data = clean_types(data)
    data.reset_index(drop=True, inplace=True)
    
    # preprocessing 2: add features
    start_time_fe = time.time()
    logger.warning("BE CAREFUL regarding features and the selected frequency!")
    logger.info("computing features ...")

    if conf['frequency'] == '1m':
        start_time_fe = time.time()
        features_engineering_1m(
            data=data,
            idx=data.index,
            stock=conf["stock_reference"]
        )
        end_time_fe = time.time()

    elif conf['frequency'] == '1h':
        start_time_fe = time.time()
        features_engineering_1h(
            data=data,
            idx=data.index,
            stock=conf["stock_reference"]
        )
        end_time_fe = time.time()
    
    elif conf['frequency'] == '1d':
        start_time_fe = time.time()
        features_engineering_1d(
            data=data,
            idx=data.index,
            stock=conf["stock_reference"]
        )
        end_time_fe = time.time()

    else:
        raise ValueError(f"define function features_engineering_{conf['frequency']}.")
    
    logger.info(f"... done in {round(end_time_fe-start_time_fe, 3)}s.")
    
    start_time_lab = time.time()
    logger.info("computing labels ...")
    optimal_actions = Labelling.compute_optimal_actions(streak_rdt=data[f"rdt_crypto_{conf['frequency']}"].values, t0=0, T=len(data), fees=0.00075)
    labels_weights_index = Labelling.compute_labels(streak_rdt=data[f"rdt_crypto_{conf['frequency']}"].values, t0=0, T=len(data), fees=conf['fees'])
    data["optimal_actions"] = optimal_actions
    data["actions_stable_to_crypto"] = labels_weights_index[0]
    data["actions_crypto_to_stable"] = labels_weights_index[1]
    data["weights_stable_to_crypto"] = labels_weights_index[2]
    data["weights_crypto_to_stable"] = labels_weights_index[3]
    data["index_knowledge_stable_to_crypto"] = labels_weights_index[4]
    data["index_knowledge_crypto_to_stable"] = labels_weights_index[5]
    logger.info(f"... done in {round(time.time()-start_time_lab, 3)}s.")
    
    # save
    start_time_saving = time.time()
    logger.info("saving data to csv ...")
    filename = f"preproc_{conf['market']}_{conf['frequency']}_f{conf['fees']}_{conf['start_date']}_to_{conf['end_date']}.csv"
    check_exists_or_create(conf['save_path'])
    save_path = os.path.join(conf['save_path'], filename)
    data.to_csv(save_path, date_format="%Y-%m-%d %H:%M:%S")
    logger.info(f"... done in {round(time.time()-start_time_saving, 3)}s.")
