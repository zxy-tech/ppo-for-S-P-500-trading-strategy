import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
import numpy as np
from pmdarima.arima import auto_arima
import yaml
with open('settings.yml') as f:
    settings = yaml.safe_load(f)

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
        if k % index_interval == 0:
            tempY = np.append(tempY, Y[k])
            date_index = np.append(date_index, Y.index[k])
    return date_index, tempY

def arimatest(historyY, predict_days):
    model = auto_arima(historyY, trace=False, error_action='ignore', suppress_warnings=True)
    model.fit(historyY)
    arima_historyY = model.predict_in_sample()
    arima_predictY = model.predict(n_periods=predict_days)
    
    return arima_historyY, arima_predictY

def main():
    raw_prices = pd.read_csv(settings['intpricesfile'], index_col=0, parse_dates=True)[settings['tickers']]
    gain = pd.read_csv(settings['intreturnsfile'], index_col=0, parse_dates=True)[settings['tickers']] + 1
    date_index = raw_prices.index
    stock_name = settings['tickers']
    index_interval = 3
    train_days = 60 
    predict_days = 3

    df_columns = []
    for i in settings['tickers']:
        for k in range(0, predict_days + 1):
            df_columns.append('{}_forecast_{}'.format(i, k))

    df_index, _ = DataExtract(raw_prices.loc[:][stock_name[0]], index_interval)
    df = pd.DataFrame(columns=df_columns, index=df_index[train_days:])

    for stock in stock_name:
        Y = raw_prices.loc[:][stock]
        date_index, Y = DataExtract(Y, index_interval)
        for i in range(train_days, len(date_index)):
            historyY = Y[i - train_days:i]
            normalizer = Normalizer(min(historyY), max(historyY))
            historyY_normalized = normalizer.normalize(historyY)
            arima_historyY, arima_predictY = arimatest(historyY_normalized, predict_days)
            # arima_predictY = normalizer.restore(arima_predictY)
            ARIMAy = arima_predictY
            arima_obs = np.insert(ARIMAy, 0, gain.loc[date_index[i]][stock])
            # print(f"arima_obs shape: {arima_obs.shape}")
            # print(f"arima_obs shape: {arima_obs}")
            temp_columns = ['{}_forecast_{}'.format(stock, k) for k in range(0, predict_days + 1)]
            df.loc[date_index[i], temp_columns] = arima_obs
    df.to_csv(settings['arima_signal'])
    return

if __name__ == "__main__":
    main()