import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests

from utils import *

class PairSelector:
    def __init__(self, data, symbols=None):
        if symbols:
            self.data = data[['Date'] + symbols]
            self.symbols = symbols
        else:
            self.data = data
            self.symbols = self.data.columns[1:]
            
    def get_all_pairs(self):
        pairs = []
        for i in range(len(self.symbols)):
            for j in range(i + 1, len(self.symbols)):
                pairs.append((self.symbols[i], self.symbols[j]))
        return pairs
    
    def get_filtered_pairs(self, num):
        all_pairs = self.get_all_pairs()
        rslt = {}
        for pair in all_pairs:
            rslt[pair] = corr(self.data[pair[0]], self.data[pair[1]])
        return list(top_n_elements(rslt, num).keys())
    
    @staticmethod
    def pair_score_func(price1, price2, method='corr'):
        if method == 'corr':
            ret1 = np.log(price1) - np.log(price1.shift(1))
            ret2 = np.log(price2) - np.log(price2.shift(1))
            return corr(ret1, ret2)
        
        elif method == 'ADF':
            adf_result = adfuller(price1 / price2)
            return -adf_result[1]
        
        elif method == 'Granger':
            ret1 = (np.log(price1) - np.log(price1.shift(1))).fillna(0)
            ret2 = (np.log(price2) - np.log(price2.shift(1))).fillna(0)
            g12_pval = grangercausalitytests(pd.DataFrame({'price1': ret1, 'price2': ret2}), 
                                             maxlag=1, verbose=False)[1][0]['ssr_chi2test'][1]
            g21_pval = grangercausalitytests(pd.DataFrame({'price2': ret1, 'price1': ret2}), 
                                             maxlag=1, verbose=False)[1][0]['ssr_chi2test'][1]
            return -(g12_pval + g21_pval)
        
        elif method == 'distance':
            cumret1 = price1 / price1.iloc[0] - 1
            cumret2 = price2 / price2.iloc[0] - 1
            return -((cumret1 - cumret2) ** 2).sum()
        
        else:
            raise NotImplementedError('method not defined!')
        
    def get_pairs_with_score(self, method='corr', num=10000):
        filtered_pairs = self.get_filtered_pairs(num=num)
        pair_with_score = {}
        for pair in filtered_pairs:
            pair_with_score[pair] = self.pair_score_func(self.data[pair[0]], self.data[pair[1]], method=method)
        return pair_with_score
    
class Backtester:
    def __init__(self, data, s1, s2, GMV=1e6):
        self.s1 = s1
        self.s2 = s2
        self.data = data[['Date', s1, s2]]
        self.GMV = GMV
        
    def simulate(self, window=50, open_score=2, close_score=0):
        self.positions = pd.DataFrame(index=self.data.index, columns=self.data.columns)
        self.positions['Date'] = self.data['Date']
        short = False
        long = False

        for t in self.data.index:
            if t < window - 1:
                continue
            recent_prices = self.data.loc[t - window + 1: t]
            
            # regression to get the spread
            log_s1 = np.log(recent_prices[self.s1])
            log_s2 = np.log(recent_prices[self.s2])
            ols = sm.OLS(log_s1, sm.add_constant(log_s2)).fit()
            
            spread = log_s1 - ols.predict(sm.add_constant(log_s2))
            mu = spread.mean()
            sigma = spread.std()
            
            total_price = recent_prices[self.s1].iloc[-1] + abs(ols.params[1]) * recent_prices[self.s2].iloc[-1]
            size_s1 = int(self.GMV / (total_price))
            size_s2 = int(self.GMV / (total_price) * ols.params[1])

            if spread.iloc[-1] > mu + open_score * sigma:
                # open short position
                short = True
                self.positions.loc[t, [self.s1, self.s2]] = [-size_s1, size_s2]
            elif spread.iloc[-1] < mu - open_score * sigma:
                # open long positions
                long = True
                self.positions.loc[t, [self.s1, self.s2]] = [size_s1, -size_s2]
            elif short and (spread.iloc[-1] < mu + close_score * sigma):
                # close short position
                short = False
                self.positions.loc[t, [self.s1, self.s2]] = [0,0]
            elif long and (spread.iloc[-1] > mu - close_score * sigma):
                # close long positions
                long = False
                self.positions.loc[t, [self.s1, self.s2]] = [0,0]

        self.positions.fillna(method='ffill', inplace=True)
        self.positions.fillna(0, inplace=True)
        
    def get_result(self):
        self.positions[self.s1 + 'trade'] = self.positions[self.s1] - self.positions[self.s1].shift(1).fillna(0)
        self.positions[self.s2 + 'trade'] = self.positions[self.s2] - self.positions[self.s2].shift(1).fillna(0)
        self.result = pd.merge(self.data, self.positions[['Date', self.s1 + 'trade', self.s2 + 'trade']], on=['Date'])
        for s in [self.s1, self.s2]:
            self.result[s + 'remainning'] = self.result[s + 'trade'].cumsum()
            self.result[s + 'pnl'] = ((-self.result[s + 'trade'] * self.result[s]).cumsum() 
                                      + self.result[s + 'remainning'] * self.result[s])
        self.result['pnl'] = self.result[self.s1 + 'pnl'] + self.result[self.s2 + 'pnl']
        self.result['daily_ret'] = (self.result['pnl'] - self.result['pnl'].shift(1).fillna(0)) / self.GMV
            
    def get_pnl(self):
        pnl = 0
        for s in [self.s1, self.s2]:
            remaining_volume = self.result[s + 'trade'].sum()
            pnl += (-self.result[s + 'trade'] * self.result[s]).sum() + remaining_volume * self.result[s].iloc[-1]
        return pnl
    
    def get_sharpe(self):
        return self.result['daily_ret'].mean() / self.result['daily_ret'].std() * np.sqrt(252)
    
    def get_notional(self):
        notional = 0
        for s in [self.s1, self.s2]:
            notional += (abs(self.result[s + 'trade'] * self.result[s])).sum()
        return notional
    
    def get_agg_trades(self):
        self.result['long'] = np.where(self.positions[self.s1] > 0, 1, 
                                       np.where(self.positions[self.s1] < 0, -1, 0))
        self.result['change'] = np.where(self.result['long'] - self.result['long'].shift(1) != 0, 1, 0)
        self.result['change'] = self.result['change'].cumsum()
        self.result['next_ret'] = self.result['daily_ret'].shift(-1).fillna(0)
        agg_dict = {
            'Date': len,
            'next_ret': np.sum
        }
        trades = self.result[self.positions[self.s1] != 0].groupby(['change']).agg(agg_dict)
        return trades
    
    def get_trading_duration(self):
        trades = self.get_agg_trades()
        return trades['Date'].mean()
    
    def get_duration_profit(self):
        trades = self.get_agg_trades()
        return trades.corr().iloc[0, 1]
            
    def plot_ret(self):
        PnL = self.result['pnl']
        plt.title('Cummulative Return')
        plt.plot(self.data['Date'], PnL / self.GMV)
        plt.show()
        
    def plot_GMV(self):
        GMV = abs(self.positions[self.s1] * self.data[self.s1]) + abs(self.positions[self.s2] * self.data[self.s2])
        plt.title('Gross Market Value (GMV)')
        plt.plot(self.data['Date'], GMV)
        plt.show()
           
class ParameterTuner:
    def __init__(self, data, pairs):
        self.data = data
        self.pairs = list(pairs)
        
    def get_best_parameter(self, pair):
        parameter_rslt = {}
        backtester = Backtester(self.data, s1=pair[0], s2=pair[1])
        for window in [30, 50, 70]:
            for open_score in [1.5, 2, 2.5]:
                for close_score in [-1, 0, 1]:
                    backtester.simulate(window=window, open_score=open_score, close_score=close_score)
                    backtester.get_result()
                    parameter_rslt[(window, open_score, close_score)] = backtester.get_sharpe()
        return list(top_n_elements(parameter_rslt, 1).keys())[0]
    
    def choose_parameters(self):
        rslt = {}
        for pair in self.pairs:
            rslt[pair] = self.get_best_parameter(pair)
        return rslt
        
class PairTrading:
    def __init__(self, pairs_with_params, data):
        self.pairs_with_params = pairs_with_params
        self.data = data
        
    def calculate_pnl(self, GMV=1e6):
        pnl_rslt = {}
        for pair in self.pairs_with_params:
            params = self.pairs_with_params[pair]
            backtester = Backtester(self.data, pair[0], pair[1], GMV=GMV)
            backtester.simulate(window=params[0], open_score=params[1], close_score=params[2])
            backtester.get_result()
            pnl_rslt[pair] = [backtester.get_pnl(), backtester.get_sharpe(), 
                              backtester.get_notional(), backtester.get_trading_duration()]
        pnl_rslt = pd.DataFrame(pnl_rslt, index=['PnL', 'Sharpe', 'Turnover', 'Trade Duration']).T.reset_index()
        pnl_rslt = pnl_rslt.rename(columns={'level_0': 'security_0', 'level_1': 'security_1'})
        pnl_rslt['return'] = pnl_rslt['PnL'] / GMV / self.data.shape[0] * 252
        pnl_rslt['Turnover'] = pnl_rslt['Turnover'] / GMV / self.data.shape[0] * 252
        display(pnl_rslt)
        
    def strategy_perf(self, GMV=1e6):
        pairs = list(self.pairs_with_params.keys())
        notional = 0
        trading_days = 0
        trades = 0
        performance = pd.DataFrame(np.zeros((self.data.shape[0], len(pairs))), columns=pairs)
        for pair in pairs:
            params = self.pairs_with_params[pair]
            backtester = Backtester(self.data, pair[0], pair[1], GMV=GMV)
            backtester.simulate(window=params[0], open_score=params[1], close_score=params[2])
            backtester.get_result()
            performance[pair] = backtester.result['daily_ret']
            notional += backtester.get_notional()
            trading_days += backtester.get_agg_trades()['Date'].sum()
            trades += backtester.get_agg_trades().shape[0]
        performance['daily_ret'] = performance.mean(axis=1)
        ann_ret = performance['daily_ret'].mean() * 252
        sharpe = performance['daily_ret'].mean() / performance['daily_ret'].std() * np.sqrt(252)
        turnver = notional / (len(pairs) * GMV) / self.data.shape[0] * 252
        trading_duration = trading_days / trades
        summary = pd.DataFrame({'ann_ret': [ann_ret], 'sharpe': [sharpe], 
                                'turnover': [turnver], 'average duration': [trading_duration]})
        display(summary)
        
    def get_sharpe(self, GMV=1e6):
        pairs = list(self.pairs_with_params.keys())
        performance = pd.DataFrame(np.zeros((self.data.shape[0], len(pairs))), columns=pairs)
        for pair in pairs:
            params = self.pairs_with_params[pair]
            backtester = Backtester(self.data, pair[0], pair[1], GMV=GMV)
            backtester.simulate(window=params[0], open_score=params[1], close_score=params[2])
            backtester.get_result()
            performance[pair] = backtester.result['daily_ret']
        performance['daily_ret'] = performance.mean(axis=1)
        ann_ret = performance['daily_ret'].mean() * 252
        sharpe = performance['daily_ret'].mean() / performance['daily_ret'].std() * np.sqrt(252)
        return sharpe