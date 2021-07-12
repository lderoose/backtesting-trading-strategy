#!/usr/bin/python3.8


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

class StrategyEvaluator:
    
    COLUMN_PORTFOLIO = 'portfolio'
    COLUMN_ASSET = 'close_norm'
    N_DEC = 6  # number of decimals
    
    def __init__(self, data, params_report):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            index must be integer, not timestamp.
        params_report : dict
            cf. strategy.conf
        """
        self.data = data
        self.params_report = params_report
        t0, T = self._get_idx()
        self.t0 = t0
        self.T = T
        
    def _get_idx(self):
        """
        get the index of start and end according to date in params report.
        """
        try:
            t0 = self.data.index[self.data.close_time == self.params_report['t0']].values[0]
            T = self.data.index[self.data.close_time == self.params_report['T']].values[0]
        except:
            print(self.data.close_time)
            raise ValueError("conf/strategy.conf t0 and T parameter be filled with any date belongs to data.")
        return t0, T
    
    def _check_exist_or_create(self, path):
        """
        if path doesn't exist, create it.
        """
        if not os.path.exists(path):
            os.mkdir(path)
        return None
        
    def _initialize_report(self, path_report):
        """
        create header of report.
        """
        s = '*' * 23
        path_folder = os.path.dirname(path_report)
        self._check_exist_or_create(path_folder)
        with open(path_report, "w") as f: 
            f.write(s + '\n')
            f.write('* STRATEGY EVALUATION *\n')
            f.write(s + '\n'*2)
            f.write(f"FREQUENCY: {self.params_report['frequency']}\n")
            f.write(f"DATERANGE: {self.params_report['t0']}  to  {self.params_report['T']} \n")
            f.write(f"BEST PARAMS SIMULATION: {self.params_report['best_params_simulation']}\n\n\n")
        return None
    
    def _add_info_to_report(self, key, value, path_report):
        """
        open and write info into .txt file.
        """
        with open(path_report, "a") as f:
            f.write(f'{key}: {value} \n')
        return None
    
    def _add_df_to_report(self, df, path_report):
        """
        write pandas.DataFrame in a .txt file.
        """
        cols = df.columns.tolist()
        index = df.index.tolist()
        header = " " * (len(index[0])+1) + ' '.join(cols) + '\n'# len(index[0]) == len(index[1])
        with open(path_report, "a") as f:
            f.write("confusion matrix:\n")
            f.write(header)
        with open(path_report, "a") as f:
            df.to_csv(path_report, header=None, index=True, sep=' ', mode='a')
        return None
    
    def _compute_diff(self):
        """
        compute strategy - asset.
        """
        return self.data[self.COLUMN_PORTFOLIO].values[self.t0:self.T] - self.data[self.COLUMN_ASSET].values[self.t0:self.T]
    
    def compute_return(self, portfolio, period=1):
        """
        compute return for each period.
        """
        if portfolio:
            ret = self.data[self.COLUMN_PORTFOLIO].diff(period) / self.data[self.COLUMN_PORTFOLIO].shift(period)
        else:
            ret = self.data[self.COLUMN_ASSET].diff(period) / self.data[self.COLUMN_ASSET].shift(period)
        ret.fillna(value=0, inplace=True)
        return ret
    
    def compute_return_interval(self, portfolio, start, end):
        """
        compute return of timeseries on given window.
        """
        if portfolio:
            ret = round(
                100 * (self.data[self.COLUMN_PORTFOLIO].values[end] - self.data[self.COLUMN_PORTFOLIO].values[start]) / \
                    self.data[self.COLUMN_PORTFOLIO].values[start], 4
                    )
        else:
            ret = round(100 * (self.data[self.COLUMN_ASSET].values[end] - self.data[self.COLUMN_ASSET].values[start]) /\
                self.data[self.COLUMN_ASSET].values[start], 4)
        return ret
    
    def compute_sharpe_ratio(self, r_p, r_f, sigma_p):
        sharpe = (r_p - r_f) / sigma_p
        # annualize : cf. book Quantitative Trading : how to build your own algorithmic trading business, p44
        if self.params_report["frequency"] == '1m':
            coeff = (60 * 24 * 365) ** 0.5
        elif self.params_report["frequency"] == '1h':
            coeff = (24 * 365) ** 0.5
        elif self.params_report["frequency"] == '1d':
            coeff = 365 ** 0.5
        else:
            raise ValueError("unknown frequency")
        sharpe *= coeff
        return round(sharpe, self.N_DEC)
    
    def display_portfolio_evolution(self, path_fig, add_switchs=True, figsize_x=20, figsize_y=5, colour='cornflowerblue'):
        """
        save graph of strategy evolution and asset evolution.
        """
        if add_switchs:
            idx_switch = self.data.index[self.data.switch == 1].values.tolist()
            max_value = max(max(self.data.portfolio.values), max(self.data.close_norm.values))
        plt.figure(figsize=(figsize_x,figsize_y))
        plt.plot(self.data[self.COLUMN_ASSET].values, label='asset norm', c='black')
        plt.plot(self.data[self.COLUMN_PORTFOLIO].values, label='portfolio', c=colour)
        if add_switchs:
            plt.vlines(x=idx_switch, ymin=0, ymax=max_value, label='switchs', linestyles='dashed')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.title('visualize switchs')
        plt.legend()
        plt.savefig(path_fig)
        return None
    
    def display_diff(self, path_fig, add_switchs=True, figsize_x=20, figsize_y=5, colour='cornflowerblue'):
        """
        save graph of strategy - asset.
        """
        diff = self._compute_diff()
        if add_switchs:
            idx_switch = self.data.index[self.data.switch == 1].values.tolist()
            max_value = max(diff)
            min_value = min(diff)
        plt.figure(figsize=(figsize_x,figsize_y))
        plt.plot(diff, label='diff', c=colour)
        plt.hlines(y=0, xmin=0, xmax=len(self.data), label='null', linestyles='-')
        if add_switchs:
            plt.vlines(x=idx_switch, ymin=min_value, ymax=max_value, label='switchs', linestyles='dashed')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.title('strategy - asset')
        plt.grid(True)
        plt.legend()
        plt.savefig(path_fig)
        return None
    
    def display_ret(self, path_fig, add_switchs=True, figsize_x=20, figsize_y=5, colour='cornflowerblue'):
        """
        save graph of return for asset and strategy.
        """
        ret_strategy = self.compute_return(portfolio=True)
        ret_asset =  self.compute_return(portfolio=False)
        if add_switchs:
            idx_switch = self.data.index[self.data.switch == 1].values.tolist()
            max_value = max(max(ret_strategy), max(ret_asset))
            min_value = min(min(ret_strategy), min(ret_asset))
        
        fig = plt.figure(figsize=(figsize_x,figsize_y))
        fig.suptitle('return of strategy & asset')
        
        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(ret_asset, label='return_asset', c=colour)
        if add_switchs:
            ax2.vlines(x=idx_switch, ymin=min_value, ymax=max_value, label='switchs', linestyles='dashed')
        ax2.set_xlabel(f'time (period: {self.params_report["frequency"]})')
        ax2.set_ylabel('value')
        ax2.grid(True)
        ax2.legend(loc='upper left')
        
        ax1 = fig.add_subplot(2,1,1, sharey=ax2)
        ax1.plot(ret_strategy, label='return_strategy', c=colour)
        if add_switchs:
            ax1.vlines(x=idx_switch, ymin=min_value, ymax=max_value, label='switchs', linestyles='dashed')
        ax1.set_ylabel('value')
        ax1.grid(True)
        ax1.legend(loc='upper left')
        
        fig.savefig(path_fig)
        return None
    
    def display_cumulative_ret(self, add_switchs, path_fig, figsize_x=20, figsize_y=5):
        """
        save graph of cumulative return for asset and strategy.
        """
        cum_ret_strategy = (1 + self.compute_return(portfolio=True)).cumprod()
        cum_ret_asset = (1 + self.compute_return(portfolio=False)).cumprod()
        if add_switchs:
            idx_switch = self.data.index[self.data.switch == 1].values.tolist()
        
        fig = plt.figure(figsize=(figsize_x,figsize_y))
        fig.suptitle('cumulative return of strategy & asset')
        
        ax1 = fig.add_subplot(2,1,2)
        ax1.plot(cum_ret_strategy, label='return_strategy', c='cornflowerblue')
        if add_switchs:
            ax1.vlines(x=idx_switch, ymin=min(cum_ret_strategy), ymax=max(cum_ret_strategy), label='switchs', linestyles='dashed')
        ax1.set_xlabel(f'time (period: {self.params_report["frequency"]})')
        ax1.set_ylabel('value')
        ax1.grid(True)
        ax1.legend(loc='upper left')
        
        ax2 = fig.add_subplot(2,1,1)
        ax2.plot(cum_ret_asset, label='return_asset', c='cornflowerblue')
        if add_switchs:
            ax2.vlines(x=idx_switch, ymin=min(cum_ret_asset), ymax=max(cum_ret_asset), label='switchs', linestyles='dashed')
        ax2.set_ylabel('value')
        ax2.grid(True)
        ax2.legend(loc='upper left')
        
        fig.savefig(path_fig)
        return None
    
    def get_confusion_matrix(self, true, pred):
        conf = pd.DataFrame(
            confusion_matrix(true, pred),
            index=['True'+str(i) for i in range(2)],
            columns = ['Pred'+str(j) for j in range(2)]
        )
        return conf
    
    def generate_report(self, path_folder):
        """
        according to params_report, generate a .txt file and save some graph in path_folder
        Parameters
        ----------
        path_folder : str
            path
        Returns
        -------
        None
        Raises
        ------
        ValueError
            if used frequency is unknown
        """
        path_report = os.path.join(path_folder, self.params_report['report_name'])
        self._initialize_report(path_report)
        
        if self.params_report['display']['diff']:
            path_file = os.path.join(path_folder, "diff.png")
            self.display_diff(
                path_fig=path_file,
                add_switchs=self.params_report['display']['switch']
                )
        
        if self.params_report['display']['evolution']:
            path_file = os.path.join(path_folder, "evolution.png")
            self.display_portfolio_evolution(
                path_fig=path_file,
                add_switchs=self.params_report['display']['switch']
            )
            
        if self.params_report['display']['return']:
            path_file = os.path.join(path_folder, "strategy_and_asset_return.png")
            self.display_ret(
                path_fig=path_file,
                add_switchs=self.params_report['display']['switch']
            )

        if self.params_report['display']['cumulative_return']:
            path_file = os.path.join(path_folder, "cumulative_return.png")
            self.display_cumulative_ret(
                path_fig=path_file,
                add_switchs=self.params_report['display']['switch']
            ) 
            
        if self.params_report['n_periods']:
            self._add_info_to_report(
                key='n_periods',
                value=len(self.data),
                path_report=path_report
            )
        if self.params_report['return']['final_return_%']:
            ret_strategy = self.compute_return_interval(
                portfolio=True,
                start=self.t0,
                end=self.T
            )
            ret_asset = self.compute_return_interval(
                portfolio=False,
                start=self.t0,
                end=self.T
            )
            self._add_info_to_report(
                key='final_return_strategy_%',
                value=ret_strategy,
                path_report=path_report
            )
            self._add_info_to_report(
                key='final_return_asset_%',
                value=ret_asset,
                path_report=path_report
            )
            
        if self.params_report['return']["mean_return_%"]:
            ret_strategy = self.compute_return(portfolio=True)
            ret_asset = self.compute_return(portfolio=False)
            
            mean_ret_strategy = round(100 * np.mean(ret_strategy), self.N_DEC)
            mean_ret_asset = round(100 * np.mean(ret_asset), self.N_DEC)
            
            self._add_info_to_report(
                key='mean_return_strategy_%',
                value=mean_ret_strategy,
                path_report=path_report
            )
            self._add_info_to_report(
                key='mean_return_asset_%',
                value=mean_ret_asset,
                path_report=path_report
            )
            
        if self.params_report['return']['std_return_%']:
            ret_strategy = self.compute_return(portfolio=True)
            ret_asset = self.compute_return(portfolio=False)
            
            std_ret_strategy = round(100 * np.std(ret_strategy), self.N_DEC)
            std_ret_asset = round(100 * np.std(ret_asset), self.N_DEC)
            
            self._add_info_to_report(
                key='std_return_strategy_%',
                value=std_ret_strategy,
                path_report=path_report
            )
            self._add_info_to_report(
                key='std_return_asset_%',
                value=std_ret_asset,
                path_report=path_report
            )
            
        if self.params_report['return']['monthly_return_%']:
            lst_return_strategy = []
            lst_return_asset = []
            
            if self.params_report['frequency'] == '1m':
                fqz = 60 * 24 * 30
            elif self.params_report['frequency'] == '1h':
                fqz = 24 * 30
            elif self.params_report['frequency'] == '1d':
                fqz = 30
            else:
                raise ValueError("unknown frequency")
            
            for i, t in enumerate(range(fqz, len(self.data), 24*30)):
                ret_strat = self.compute_return_interval(
                    portfolio=True,
                    start=t - fqz,
                    end=t
                    )   
                ret_asset = self.compute_return_interval(
                    portfolio=False,
                    start=t - fqz,
                    end=t
                    )
                tupl_strat = (f'month {i} to {i+1}', ret_strat) 
                tupl_ass = (f'month {i} to {i+1}', ret_asset)
                lst_return_strategy.append(tupl_strat)
                lst_return_asset.append(tupl_ass)
            
            self._add_info_to_report(
                key='monthly_return_strategy_%',
                value=", ".join(map(str, lst_return_strategy)),
                path_report=path_report
            )
            self._add_info_to_report(
                key='monthly_return_asset_%',
                value=", ".join(map(str, lst_return_asset)),
                path_report=path_report
            )
                
        if self.params_report['return']['annualy_retun_%']:
            lst_return_strategy = []
            lst_return_asset = []
            
            if self.params_report['frequency'] == '1m':
                fqz = 60 * 24 * 30 * 12
            elif self.params_report['frequency'] == '1h':
                fqz = 24 * 30 * 12
            elif self.params_report['frequency'] == '1d':
                fqz = 30 * 12
            else:
                raise ValueError("unknown frequency")
            
            for i, t in enumerate(range(fqz, len(self.data), fqz)):
                ret_strat = self.compute_return_interval(
                    portfolio=True,
                    start=t - fqz,
                    end=t
                    )   
                ret_asset = self.compute_return_interval(
                    portfolio=False,
                    start=t - fqz,
                    end=t
                    )
                tupl_strat = (f'year {i} to {i+1}', ret_strat) 
                tupl_ass = (f'year {i} to {i+1}', ret_asset)
                lst_return_strategy.append(tupl_strat)
                lst_return_asset.append(tupl_ass)
            
            self._add_info_to_report(
                key='annualy_return_strategy_%',
                value=", ".join(map(str, lst_return_strategy)),
                path_report=path_report
            )
            self._add_info_to_report(
                key='annualy_return_asset_%',
                value=", ".join(map(str, lst_return_asset)),
                path_report=path_report
            )
            
        if self.params_report['diff_metrics']:
            diff = self._compute_diff()
            min_diff = min(diff)
            max_diff = max(diff)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            
            self._add_info_to_report(
                key='min_diff',
                value=min_diff,
                path_report=path_report
            )
            self._add_info_to_report(
                key='max_diff',
                value=max_diff,
                path_report=path_report
            )
            self._add_info_to_report(
                key='mean_diff',
                value=mean_diff,
                path_report=path_report
            )
            self._add_info_to_report(
                key='std_diff',
                value=std_diff,
                path_report=path_report
            )

        if self.params_report["sharpe_ratio"]:
            ret_strategy = self.compute_return(portfolio=True)
            ret_asset = self.compute_return(portfolio=False)
            
            mean_ret_strategy = np.mean(ret_strategy)
            mean_ret_asset = np.mean(ret_asset)
            std_ret_strategy = np.std(ret_strategy)
            std_ret_asset = np.std(ret_asset)
            
            sharpe_ratio_strategy_stable = self.compute_sharpe_ratio(
                r_p=mean_ret_strategy,
                r_f=0,
                sigma_p=std_ret_strategy
            )
            sharpe_ratio_strategy_asset = self.compute_sharpe_ratio(
                r_p=mean_ret_strategy,
                r_f=mean_ret_asset,
                sigma_p=std_ret_strategy
            )
            sharpe_ratio_asset_stable = self.compute_sharpe_ratio(
                r_p=mean_ret_asset,
                r_f=0,
                sigma_p=std_ret_asset
            )
            
            self._add_info_to_report(
                key='annual_sharpe_ratio_strategy_with_stable',
                value=sharpe_ratio_strategy_stable,
                path_report=path_report
            )
            self._add_info_to_report(
                key='annual_sharpe_ratio_strategy_with_asset',
                value=sharpe_ratio_strategy_asset,
                path_report=path_report
            )
            self._add_info_to_report(
                key='annual_sharpe_ratio_asset_with_stable',
                value=sharpe_ratio_asset_stable,
                path_report=path_report
            )
        
        if self.params_report['actions_metrics']['accuracy']:
            accuracy = round(accuracy_score(self.data.optimal_actions.values, self.data.switch.values), self.N_DEC)
            self._add_info_to_report(
                key='actions_accuracy',
                value=accuracy,
                path_report=path_report
            )
            
        if self.params_report['actions_metrics']['precision']:
            precision = round(precision_score(self.data.optimal_actions.values, self.data.switch.values), self.N_DEC)
            self._add_info_to_report(
                key='actions_precision',
                value=precision,
                path_report=path_report
            )
            
        if self.params_report['actions_metrics']['precision']:
            recall = round(recall_score(self.data.optimal_actions.values, self.data.switch.values), self.N_DEC)
            self._add_info_to_report(
                key='actions_recall',
                value=recall,
                path_report=path_report
            )
            
        if self.params_report['actions_metrics']['confusion_matrix_actions']:
            conf_actions = self.get_confusion_matrix(self.data.optimal_actions.values, self.data.switch.values)
            self._add_df_to_report(
                df=conf_actions,
                path_report=path_report
            )
            
        return None
