import numpy as np
import pandas as pd
import gym
# import gym.spaces
import torch
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt 
# from gym.utils import seeding
from datetime import timedelta
EPS = 1e-8


def spyProcess():
    df = pd.read_csv("original_data/SPY_original.csv",  index_col=0, parse_dates=True)
    sff = pd.read_csv("PV/SFFRL.csv",  index_col=0, parse_dates=True)
    df = df.loc[sff.index]
    returns = np.log(df/df.shift(1))
    returns.fillna(0, inplace=True)
    returns.to_csv("PV/SPY.csv")
    return 


def calSharpe():
    #df = pd.read_csv("SFF_method.csv")['SFF']
    files = ['SFFRL', 'UCRP', 'OLMAR', 'WMAMR', 'MPT']
    spy =  pd.read_csv("PV/SPY.csv", index_col=0, parse_dates=True)['SPY']
    for file in files:
        df = pd.read_csv("PV/"+file+".csv", index_col=0, parse_dates=True)[file]

        TRADING_DAYS = df.shape[0]
        returns = np.log(df/df.shift(1))
        #print(returns)
        returns.fillna(0, inplace=True)
        volatility = returns.std()
        sharpe_ratio = (returns-spy).mean()/volatility
        print(file, ": ", sharpe_ratio)

# spyProcess()
calSharpe()

