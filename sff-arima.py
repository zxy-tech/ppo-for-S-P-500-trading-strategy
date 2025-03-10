import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pmdarima.arima import auto_arima
from math import sqrt
from sklearn.metrics import mean_squared_error
import sff_part
import yaml

with open('settings.yml') as f:
    settings = yaml.safe_load(f)


def arimatest(historyY, predict_days):
    model = auto_arima(historyY, trace=False, error_action='ignore', suppress_warnings=True)
    model.fit(historyY)
    arima_historyY = model.predict_in_sample()
    arima_predictY = model.predict(n_periods=predict_days)
    
    #history_rmse = sqrt(mean_squared_error(arima_historyY, historyY))
    #predict_rmse = sqrt(mean_squared_error(arima_predictY, predictY))

    #print("history rmse is {}, predict rmse is {}", format(history_rmse, predict_rmse))
    return arima_historyY, arima_predictY#, history_rmse, predict_rmse


    
def SMAPE(x, y):
    n = x.shape[0]
    return (2/n)*(np.abs(x-y)/(np.abs(x)+np.abs(y))).sum()

def main():
    raw_prices = pd.read_csv(settings['intpricesfile'], index_col=0, parse_dates=True)
    date_index = raw_prices.index
    stock_name = raw_prices.columns
    stock_name = ['C', 'AAPL', 'WFC']
    err_threshold = 0.001 #[0.4, 0.3, 0.2, 0.1, 0.05, 0.01];
    train_days =100
    predict_days = 10


    SFFModel = sff_part.SineFunctionFitting()
    for stock in stock_name:
        Y = raw_prices.loc[:][stock]
        X = np.array(range(0, Y.shape[0]))
        SFFy = []
        ARIMAy = []
        ACTUALy = []
        for start_date in [2500]:#[570, 575, 150, 70, 75]: #0 150 120 140 210(not very good)
            historyY = Y[start_date:start_date + train_days]
            historyX = X[start_date:start_date + train_days]
            normalizer = sff_part.Normalizer(min(historyY),max(historyY))
            historyY = normalizer.normalize(historyY)    
            predictY = Y[start_date + train_days:start_date + train_days + predict_days]
            predictX = X[start_date + train_days:start_date + train_days + predict_days]
            SFF_LearningSequence, SFF_Para, SFF_LearnErr = SFFModel.SineFitting(historyX,historyY,err_threshold,1000)
            SFFPredictY = SFFModel.SFFPredict(SFF_Para, predictX)
            SFFPredictY = normalizer.restore(SFFPredictY)
            arima_historyY, arima_predictY = arimatest(historyY, predict_days)
            arima_predictY = normalizer.restore(arima_predictY)
            SFFy = np.append(normalizer.restore(SFF_LearningSequence), SFFPredictY)
            ARIMAy = np.append(normalizer.restore(arima_historyY), arima_predictY)
            ACTUALy = np.append(normalizer.restore(historyY), predictY)
            date_index = raw_prices.index[start_date+1:start_date+train_days+predict_days]
            print(raw_prices.index[start_date:start_date + train_days])
            print(raw_prices.index[start_date + train_days:start_date + train_days + predict_days])
            combine = 0.7*SFFy + 0.3*ARIMAy
            print("sff rmse: ", sqrt(mean_squared_error(SFFy[train_days-1:], ACTUALy[train_days-1:])))
            print("arima rmse: ", sqrt(mean_squared_error(ARIMAy[train_days-1:], ACTUALy[train_days-1:])))
            print("combine rmse: ", sqrt(mean_squared_error(combine[train_days-1:],ACTUALy[train_days-1:])))
            print("sff smape: ", SMAPE(SFFy[train_days-1:], ACTUALy[train_days-1:]))
            print("arima rmse: ", SMAPE(ARIMAy[train_days-1:],ACTUALy[train_days-1:]))
            print("combine rmse: ", SMAPE(combine[train_days-1:], ACTUALy[train_days-1:]))
            plt.figure(num=1, figsize=(9, 5))
            plt.plot(date_index,ACTUALy[1:], label = 'Real')
            plt.axvline(x=date_index[train_days-2], color='k', linestyle='--')
            plt.plot(date_index,ARIMAy[1:], '-.' ,label = 'ARIMA')
            #plt.plot(historyX, normalizer.restore(historyY), label = 'Actual History Value')
            #plt.plot(historyX, normalizer.restore(SFF_LearningSequence), label = 'SFF_LearningSequence')
            plt.plot(date_index,SFFy[1:], '-.' ,label = 'SFF')
            plt.plot(date_index,combine[1:], '--' ,label = 'SFF-ARIMA')
            plt.xlabel('date')
            plt.ylabel('stock price')
            plt.legend()
            plt.grid()
            plt.tight_layout()            #plt.savefig('plot/SFFvsARIMA_{}_{}.png'.format(stock,start_date))
            plt.savefig('SFF-ARIMA_plot/SFFvsARIMA_{}_{}.pdf'.format(stock,start_date))
            plt.clf()
    return

def GernerageSFFARIMASignals():
    sff_prices = pd.read_csv(settings['sff_price'], index_col = 0, parse_dates=True)
    sff_signals = pd.read_csv(settings['sff_signal'], index_col=0, parse_dates=True)
    arima_signals = pd.read_csv(settings['arima_signal'], index_col = 0, parse_dates=True)
    gain = pd.read_csv(settings['sff_gain'], index_col = 0, parse_dates=True)
    sff_signals = sff_signals.loc[arima_signals.index]
    signals = 0.7*sff_signals + 0.3*arima_signals
    gain = gain.loc[arima_signals.index]
    sff_prices = sff_prices.loc[arima_signals.index]
    signals.to_csv(settings['signals_file'])
    gain.to_csv(settings['gain_file'])
    sff_prices.to_csv(settings['prices_file'])

# GernerageSFFARIMASignals()
main()
