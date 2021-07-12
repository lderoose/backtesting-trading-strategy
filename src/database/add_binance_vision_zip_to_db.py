#!/usr/bin/python3.8


import os
import re
import shutil
import zipfile
from calendar import monthrange

import pandas as pd
from tqdm import tqdm
from pyhocon import ConfigFactory

from utils.conf_logger import configure_logger
from cl_binance_database import BinanceDatabase


# functions
def get_lst_path_files(path_historical_klines):
    """
    extract all the zip file belong to path_historical_klines.
    Args:
        path_historical_klines (str): root folder
    Returns:
        [list]: path of all the zip files
    """
    lst_path_files = []
    for root, _, files in os.walk(path_historical_klines, topdown=False):
        for f in files:
            lst_path_files.append(os.path.join(root, f))
    return lst_path_files


def unzip(path_zip, path_save):
    """
    unzip the file and save it.
    """
    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(path_save)
    return None


def _get_n_days(year, month):
    """
    compute the number of days of the 'month' for this 'year'.
    Args:
        year (int): year
        month (int): number of the month
    Returns:
        [int]: number of days
    """
    return monthrange(year, month)[1]


def _get_year_month(path_file):
    """
    extract the year and the month from the name of the path.
    Args:
        path_file (str): path of the file.
    Returns:
        [tuple(int, int)]: year, month
    """
    year = int(re.findall(r"-(\d{4,})-", path_file)[0])
    month = int(re.findall(r"-(\d{2,})\.", path_file)[0])
    return year, month


def get_market(path_file, path_save, frequency):
    market = re.findall(path_save + r"/(.*)-" + frequency, path_file)[0]
    return market


def format_to_insert(s):
    s = ', '.join(['"' + str(e) + '"' for e in s.split(',')])
    return s


def _report_error(path_file, n_errors, path_report):
    """
    fill a .txt for report files where numbers of rows is incorrect.
    Args:
        path_file (str): path of the file. 
        n_errors (int): number of erros.
        path_report (str): Defaults to PATH_REPORT.
    """
    with open(path_report, 'a') as r:
        r.write(f'{path_file} : {n_errors}\n')
    return None


def check_data_rows(path_file, frequency, path_report):
    """
    check if the number of data scrapped is correct or not. If not, report in a .txt file the name of
    the file and the number of missing rows.
    Args:
        path_file (str): path of the file. 
        frequency (str, optional): Defaults to "1m".
    Returns:
        [bool]: incorrect or not
    """
    correct_n_rows = False
    with open(path_file, 'r') as file:
        n_rows = sum(1 for row in file)

    # compute true number of rows
    year, month = _get_year_month(path_file)
    n_days = _get_n_days(year, month)
    if frequency == "1m":
        true_n_rows = 60 * 24 * n_days
    elif frequency == "1h":
        true_n_rows = 24 * n_days
    elif frequency == "1d":
        true_n_rows = n_days

    # check if correct
    if true_n_rows == n_rows:
        correct_n_rows = True
    else:
        n_errors = true_n_rows - n_rows
        _report_error(path_file=path_file, n_errors=n_errors, path_report=path_report)
    return correct_n_rows


if __name__ == "__main__":
    # settings
    LST_COLUMNS = [
        "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
        "taker_buy_quote_asset_volume", "ignore"
        ]
    conf = ConfigFactory.parse_file("conf/database.conf")
    logger = configure_logger(filename=os.path.basename(__file__))
    
    # create connection and table if no exists
    db = BinanceDatabase(
        database=conf["database"],
        table=conf["table"],
        logger=configure_logger(filename='database_logger')
    )
    db.create_table()
    logger.debug("connection do database done.")

    lst_path_files = get_lst_path_files(path_historical_klines=conf['path_historical_klines'])
    
    logger.info("start open zip and add to database ...")
    for path_file in tqdm(lst_path_files):
        unzip(path_zip=path_file, path_save=conf['path_save'])
        path_file_unzip = os.path.join(conf['path_save'], os.path.basename(path_file)).replace("zip", "csv")
        market = get_market(
            path_file=path_file_unzip,
            frequency=conf['frequency'],
            path_save=conf['path_save']
            )
        correct_file = check_data_rows(
            path_file=path_file_unzip,
            path_report=conf['path_report'],
            frequency=conf['frequency']
            )

        data = pd.read_csv(path_file_unzip, header=None)
        data.columns = LST_COLUMNS
        open_time_human = pd.to_datetime(data["open_time"], unit='ms', utc=False).dt.strftime('%Y-%m-%d_%H:%M:%S')
        data.insert(
            loc=0,
            column="new_0",
            value=[market for _ in range(data.shape[0])]
            )
        data.insert(
            loc=1,
            column="new_1",
            value=[conf['frequency'] for _ in range(data.shape[0])]
            )
        data.insert(
            loc=data.shape[1],
            column="new_2",
            value=[v for v in open_time_human]
        )
        db.insert_in_db(data.values)

    db.close_connection()
    logger.info("... done.")
    
    logger.info("deleting zip files ...")
    shutil.rmtree(conf['path_save'])
    logger.info("... done.")
