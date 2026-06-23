import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
plt.style.use("seaborn-v0_8")
class MeanRev_bt():
    ''' Class for the vectorized backtesting of Bollinger Bands-based trading strategies.
    '''    
    def __init__(self, symbol, SMA, dev, start, end, tc = 0):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        SMA: int
            moving window in bars (e.g. days) for SMA
        dev: int
            distance for Lower/Upper Bands in Standard Deviation units
        start: str
            start date for data import
        end: str
            end date for data import
        tc: float
            proportional transaction/trading costs per trade
        '''
        
        self._symbol = symbol
        self._SMA = SMA
        self._dev = dev
        self._start = start
        self._end = end
        self._tc = tc
        self.get_data()
        self.prepare_data()

        
    def __repr__(self):
        return "Mean Reversion Backtesting symbol = {} with SMA = {}, dev = {}, start = {}, end = {}, tc = {}".format(self._symbol, self._SMA, self._dev, self._start, self._end, self._tc)

    def get_data(self):
        ''' Imports the data from intraday_pairs.csv (source can be changed).
        '''
        raw = pd.read_csv("intraday_pairs.csv", index_col = "time", parse_dates = ["time"])
        raw = raw.loc[self._start : self._end, self._symbol].to_frame()
        raw.rename(columns = {self._symbol : "price"}, inplace = True)
        raw["returns"] = np.log(raw.price.div(raw.price.shift(1)))
        raw["creturns"] = raw.returns.cumsum().apply(np.exp)
        self.data = raw.copy().dropna()

    def prepare_data(self):
        '''Prepares the data for strategy backtesting (strategy-specific).
        '''
        data = self.data.copy()
        data["SMA"] = data.price.rolling(self._SMA).mean()
        data["Upper"] = data.SMA + data.price.rolling(self._SMA).std()*self._dev
        data["Lower"] = data.SMA - data.price.rolling(self._SMA).std()*self._dev
        self.data = data.copy().dropna()


    def set_parameters(self, SMA = None, dev = None):
        ''' Updates parameters (SMA, dev) and the prepared dataset.
        '''
        if SMA is not None:
            self._SMA = SMA
            self.data["SMA"] = self.data.price.rolling(self._SMA).mean()
            self.data["Upper"] = self.data.SMA + self.data.price.rolling(self._SMA).std()*self._dev
            self.data["Lower"] = self.data.SMA - self.data.price.rolling(self._SMA).std()*self._dev
        if dev is not None:
            self._dev = dev 
            self.data["Upper"] = self.data.SMA + self.data.price.rolling(self._SMA).std()*self._dev
            self.data["Lower"] = self.data.SMA - self.data.price.rolling(self._SMA).std()*self._dev
        
    def test_strategy(self):
        ''' Backtests the Bollinger Bands-based trading strategy.
        '''
        data = self.data.copy()
        data["position"] = np.where(data.price < data.Lower, 1, np.nan) # open Long if price < Lower band (OverSold)
        data["position"] = np.where(data.price > data.Upper, -1, data["position"]) # open Short if price > Upper band (OverBought)
            
        data["distance"] = data.price - data.SMA
        data["position"] = np.where(data.distance * data.distance.shift(1) < 0, 0, data["position"])
        data["position"] = data.position.ffill().fillna(0)

        data["strategy"] = data.returns * data.position.shift(1)
        data["trades"] = data.position.diff().abs()
        data["cstrategy"] = data.strategy.cumsum().apply(np.exp) - data.trades*self._tc
        self.result = data.copy().dropna()
        
        perf = data.cstrategy.iloc[-1]
        out_perf = perf - data.creturns.iloc[-1]
        return round(perf, 3), round(out_perf, 3)
        
    def plot_result(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.result is None:
            print("Please run test_strategy() first!")
        else:
            data = self.result.copy()
            data[["creturns", "cstrategy"]].plot(figsize = (15,8))
            plt.title("{} | SMA = {} | dev = {} | tc = {}".format(self._symbol, self._SMA, self._dev, self._tc), fontsize = 12)
            plt.show()
            

    def optimize_strat(self, SMA_range, dev_range):
        ''' Finds the optimal strategy (global maximum) given the Bollinger Bands parameter ranges.

        Parameters
        ----------
        SMA_range, dev_range: tuple
            tuples of the form (start, end, step size)
        '''
        comb = list(product(range(*SMA_range), range(*dev_range)))
        result = []

        for i in comb:
            self.set_parameters(i[0], i[1])
            result.append(self.test_strategy()[0])

        best_perf = np.max(result)
        opt = comb[np.argmax(result)]

        #Set/Run optimize strategy
        self.set_parameters(opt[0], opt[1])
        self.test_strategy()

        many_results = pd.DataFrame(data = comb, columns = ["SMA", "dev"])
        many_results["performance"] = result
        self.result_overview = many_results

        return opt, best_perf
        



















    