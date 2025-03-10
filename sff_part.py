import os
import numpy as np
from numpy import *
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
from scipy.optimize import leastsq
import yaml
with open('settings.yml') as f:
    settings = yaml.safe_load(f)

class SineFunctionFitting:
    def __init__(self):
        super().__init__

    def SinFunFix(self, Co, x):
        return np.multiply(Co[0], np.sin(np.multiply(Co[1], x) + Co[2]))

    def ErrFun(self, Co, x, y):
        return y-self.SinFunFix(Co, x)

    def SineFitting(self, X, Y, ErrThreshold, fitTimes):
        LearnNum = Y.shape[0]
        LearningSequence = np.zeros(LearnNum)

        # Set the initial value of e
        LearnErr = Y
        MinLearnErr = np.inf

        # Set the maximum number of sine functions n_max to 100
        n_max = 100

        amp = np.zeros(n_max)
        # Parameters of sine functions
        Para = np.zeros([1,3])
        
        # Use sine function to fit e
        for i in range(0, n_max):
            print(MinLearnErr)
            amp[i] = 2*np.max(np.abs(LearnErr))
            zynmin = np.inf
            # Learning sequence L_i
            L=np.zeros(X.shape[0]);
            for zynIndex in range(0, fitTimes):
                # According to least square principle
                IntiParameter = np.array([amp[i]*(np.random.rand(1)-0.5),10*(np.random.rand(1)-0.5),2*np.pi*(np.random.rand(1)-0.5)])
                tempCo = leastsq(self.ErrFun, IntiParameter, args=(X, LearnErr), maxfev=2000)
                tempCo = tempCo[0]
                # Lt is used to calculate the learning sequence
                Lt = self.SinFunFix(tempCo, X)
                TempLearnErr = np.linalg.norm(LearnErr-Lt)/np.sqrt(LearnNum)
                if TempLearnErr < zynmin:
                    zynmin=TempLearnErr
                    Co = [tempCo]
                    L = Lt                 

            # Update LearnErr
            LearnErr = LearnErr-L
            LearningSequence = LearningSequence+L
            TempErr = np.linalg.norm(LearnErr)/np.sqrt(LearnNum)
            Para = np.append(Para, Co, axis=0)
            if (MinLearnErr > TempErr):
                MinLearnErr = TempErr
            if (MinLearnErr < ErrThreshold):
                break
        Para = Para[1:Para.shape[0]]
        return LearningSequence, Para, LearnErr
    
    def SFFPredict(self, Para, PredictX):
        PredictY = np.zeros(PredictX.shape)
        for j in range(Para.shape[0]):
            PredictY = np.add(PredictY, self.SinFunFix(Para[j], PredictX)) 
        return PredictY

class Normalizer:
    def __init__(self, minY, maxY):
        self.maxY = maxY
        self.minY = minY

    def normalize(self, Y):
        return (Y-self.minY)/(self.maxY-self.minY)

    def restore(self, normY):
        return normY*(self.maxY-self.minY)+self.minY


def DataExtract(Y, index_interval):
    tempY = []
    date_index = []
    for k in range(Y.shape[0]):
        if k%index_interval == 0:
            tempY = np.append(tempY, Y[k])
            date_index = np.append(date_index, Y.index[k])
    return date_index, tempY


def main():
    #raw_prices = pd.read_csv(os.path.join(os.path.dirname(__file__), 'prices.csv'), index_col=0, parse_dates=True)
    raw_prices = pd.read_csv(settings['intpricesfile'], index_col=0, parse_dates=True)[settings['tickers']]
    gain = pd.read_csv(settings['intreturnsfile'], index_col=0,parse_dates=True)[settings['tickers']]+1
    date_index = raw_prices.index
    stock_name = settings['tickers']

    err_threshold = 0.05 #[0.4, 0.3, 0.2, 0.1, 0.05, 0.01];
    index_interval = 3
    train_days = 60 
    predict_days = 3

    df_columns = []
    for i in settings['tickers']:
        for k in range(0, predict_days+1):
            df_columns = np.append(df_columns, '{}_forecast_{}'.format(i,k))

    df_index, _ = DataExtract(raw_prices.loc[:][stock_name[0]], index_interval)
    df = pd.DataFrame(columns=df_columns, index=df_index[train_days:])

    SFFModel = SineFunctionFitting()
    for stock in stock_name:
        Y = raw_prices.loc[:][stock]
        X = np.array(range(0, Y.shape[0]))
        date_index, Y = DataExtract(Y, index_interval)
        for i in range(train_days, date_index.shape[0]):
            print(stock, i)
            historyY = Y[i-train_days:i]
            historyX = X[0:train_days]
            normalizer = Normalizer(min(historyY),max(historyY))
            historyY = normalizer.normalize(historyY)    
            predictY = Y[i:i+predict_days]
            predictX = X[train_days:train_days+predict_days]
            SFF_LearningSequence, SFF_Para, SFF_LearnErr = SFFModel.SineFitting(historyX,historyY,err_threshold,1000)
            SFFPredictY = SFFModel.SFFPredict(SFF_Para, predictX)
            SFFPredictY = normalizer.restore(SFFPredictY)
            SFFPredictGain = SFFPredictY/normalizer.restore(historyY[-1])
            SFFObs = np.insert(SFFPredictGain, 0, gain.loc[date_index[i]][stock])
            temp_columns = []
            for k in range(0, predict_days+1):
                temp_columns = np.append(temp_columns, '{}_forecast_{}'.format(stock,k))
            df.loc[date_index[i]][temp_columns] = SFFObs
        #plt.plot(historyX, normalizer.restore(historyY), label = 'Actual History Value')
        #plt.plot(historyX, normalizer.restore(SFF_LearningSequence), label = 'SFF_LearningSequence')
        #plt.plot(np.append(historyX,predictX), np.append(normalizer.restore(historyY), predictY), label = 'Actual Value')
        #plt.plot(np.append(historyX,predictX), np.append(normalizer.restore(SFF_LearningSequence), SFFPredictY), label = 'SFF_testSeq')
        #plt.legend()
        #plt.savefig('plot/SFF_{}.png'.format(stock))
        #plt.clf()
    df.to_csv(settings['sff_signal'])
    return


def PriceGainProcess():
    with open('settings.yml') as f:
        settings = yaml.safe_load(f)

    #raw_prices = pd.read_csv(os.path.join(os.path.dirname(__file__), 'prices.csv'), index_col=0, parse_dates=True)
    raw_prices = pd.read_csv(settings['intpricesfile'], index_col=0, parse_dates=True)[settings['tickers']]
    date_index = raw_prices.index
    stock_name = settings['tickers']

    index_interval = 3
    train_days = 60 
    predict_days = 3


    df_index, _ = DataExtract(raw_prices.loc[:][stock_name[0]], index_interval)
    df_index = df_index[train_days:]
    df_price = pd.DataFrame(columns=raw_prices.columns, index=df_index)
    df_gain = pd.DataFrame(columns=raw_prices.columns, index=df_index[1:])


    df_price = raw_prices.loc[df_index][stock_name].copy()
    df_gain = df_price[1:][:].values/df_price[0:-1][:].values

    df_gain = pd.DataFrame(df_gain, index=df_index[1:], columns=raw_prices.columns)

    df_price.iloc[1:,:].to_csv(settings['sff_price'])
    df_gain.to_csv(settings['sff_gain'])

# PriceGainProcess()
# main()

