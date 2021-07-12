#!/usr/bin/python3.8


import numpy as np
from numba import njit


class TimeSeriesFeaturesEngineering:
    
    @staticmethod
    def expected_return(timeseries, period):
        """
        compute expected return given period.
        t(h) - t(h-period) / t(h-period)
        Parameters
        ----------
        timeseries : pandas.Series
        period : int
            must be > 0
        Returns
        -------
        pandas.Series
            expected return
        """
        r = timeseries.diff(period) / timeseries.shift(period)
        r.fillna(value=0, inplace=True)
        return r
    
    @staticmethod
    @njit()
    def compute_volatility(timeseries, window, freq):
        """
        compute dynamic volatility : std of expected return given window and freq.
        Parameters
        ----------
        timeseries : pandas.Series or numpy.ndarray
            expected return
        window : int
        freq : int
        Returns
        -------
        numpy.ndarray
            volatility at each time
        """
        array_volatility = np.empty(shape=len(timeseries), dtype=np.float64)
        
        for i in range(len(timeseries)):
            if i >= window*freq:
                sl = slice(i+freq-(window*freq), i+1, freq)
                array_volatility[i] = np.std(timeseries[sl])
            else:
                sl = slice(i%freq, i+1, freq)
                array_volatility[i] = np.std(timeseries[sl])    
        return array_volatility

    @staticmethod
    @njit()
    def compute_cumulative_expected_return(timeseries, window, freq):
        """
        cumprod of expected return given timerseires and freq.
        Parameters
        ----------
        timeseries : pandas.Series or numpy.ndarray
            expected return
        window : int
        freq : int
        Returns
        -------
        numpy.ndarray
        """
        array_cum_expect_ret = np.empty(shape=len(timeseries), dtype=np.float64)
        
        for i in range(len(timeseries)):
            if i >= window*freq:
                sl = slice(i+freq-(window*freq), i+1, freq)
                array_cum_expect_ret[i] = np.cumprod(1 + timeseries[sl])[-1] 
            else:
                sl = slice(i%freq, i+1, freq)
                array_cum_expect_ret[i] = np.cumprod(1 + timeseries[sl])[-1]   
        return array_cum_expect_ret

    @staticmethod
    @njit(parallel=True)
    def compute_length_streak(timeseries, freq, positive=1):
        """
        since how many times streak is strictly rising (or decreasing) for each time.
        Parameters
        ----------
        timeseries : pandas.Series 
            expected return
        freq : int
        positive : int, optional
            rising = 1, decreasing = 0, by default 1
        Returns
        -------
        numpy.ndarray
        """
        int_streak = (timeseries > 0).astype(np.uint64)
        array_length_streak = np.empty(shape=len(timeseries), dtype=np.uint64)
        
        for i in range(len(timeseries)):
            v = int_streak[i]
            if positive:
                if v:
                    cpt = 1
                else:
                    cpt = 0
            else:
                if not v:
                    cpt = 1
                else:
                    cpt = 0
                    
            for j in range(i-freq, i%freq-1, -freq):
                previous_v = int_streak[j]
                if previous_v == v == positive:
                    cpt += 1
                else:
                    break
            array_length_streak[i] = cpt    
        return array_length_streak

    @staticmethod
    @njit()
    def window_movement_counter(timeseries, window, positive=1):
        """
        given a window count how many increase (resp. decrease) have been.
        Parameters
        ----------
        timeseries : pandas.Series
            expected return
        window : int
        positive : int, optional
            rising = 1, decreasing = 0, by default 1
        Returns
        -------
        numpy.ndarray
        """
        array_movment_counter = np.empty(shape=len(timeseries), dtype=np.uint64)
        
        if positive:
            int_streak = (timeseries > 0).astype(np.uint64)
        else:
            int_streak = (timeseries <= 0).astype(np.uint64)
            
        for i in range(len(timeseries)):
            if i > window:
                array_movment_counter[i] = np.sum(int_streak[(i+1-window):(i+1)])
            else:
                array_movment_counter[i] = np.sum(int_streak[:(i+1)])
        return array_movment_counter
