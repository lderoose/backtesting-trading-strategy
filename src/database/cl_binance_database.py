#!/usr/bin/python3.8


import sqlite3
from sqlite3 import Error


class BinanceDatabase:
    """
        Manage database creation and insertion.
    """

    def __init__(self, database, table, logger):
        """
        Args:
            database (str): name of the database.
            table (str): name of the table.
        """
        self.database = database
        self.table = table
        self.conn = sqlite3.connect(self.database)
        self.__logger = logger

    def close_connection(self):
        """
        close connection to SQL database.
        Returns:
            None: None
        """
        try:
            self.conn.close()
            self.__logger.debug("connection was closed")
        except Error as e:
            self.__logger.error(e)
        return None

    def create_table(self):
        """
        create table if no exists.
        Returns:
            None: None
        """
        try:
            c = self.conn.cursor()
            request = "CREATE TABLE IF NOT EXISTS '" + self.table + "'(\
                            market VARCHAR(20),\
                            frequency VARCHAR(10),\
                            open_time LONG,\
                            open FLOAT,\
                            high FLOAT,\
                            low FLOAT,\
                            close FLOAT,\
                            volume FLOAT,\
                            close_time LONG,\
                            quote_asset_volume FLOAT,\
                            number_of_trades INTEGER,\
                            taker_buy_base_asset_volume FLOAT,\
                            taker_buy_quote_asset_volume FLOAT,\
                            ignore FLOAT,\
                            open_time_human DATETIME,\
                            PRIMARY KEY(\
                                market,\
                                frequency,\
                                open_time,\
                                close_time)\
                            );"
            c.execute(request)
            self.__logger.debug("SQL table was created successfully.")
        except:
            self.__logger.debug("SQL table already exists, no creation.")
        return None

    def execute_sql_request(self, request):
        """
        execute the SQL request given in parameter.
        Args:
            request (str): SQL request
        Returns:
            None: None
        """
        try:
            c = self.conn.cursor()
            c.execute(request)
            rows = c.fetchall()
            return rows
        except Error as e:
            self.__logger.error(e)
        return None

    def insert_in_db(self, values, method='many'):
        """
        insert values in SQL table.
        Args:
            values (str): must contains all the values of table's column with order.
        Returns
            None: None
        """
        try:
            c = self.conn.cursor()
            try:
                request_insert = 'INSERT INTO "' + self.table + \
                    '" VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'
                if method == "many":
                    c.executemany(request_insert, values)
                else:
                    c.execute(request_insert, values)
            except sqlite3.IntegrityError:
                # data already exists in database
                self.__logger.debug(
                    f"{values} data already exist in database.")
                pass
            self.conn.commit()
        except Error as e:
            self.__logger.error(e, values)
        return None
