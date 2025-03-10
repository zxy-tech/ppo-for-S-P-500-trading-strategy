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

def sinefig():
    testX = np.linspace(-3.14,3.14,100)

    plt.plot(testX, np.sin(testX))
    plt.plot(testX, 0.8*np.sin(testX-2))
    plt.plot(testX, 0.4*np.sin(1.5*testX-2))
    plt.plot(testX, 0.4*np.sin(1.5*testX))

    plt.savefig('sine.png')
    return

def stockfig():
    prices_file = 'SFFRL_data/price.csv' #'prices.csv

    prices = pd.read_csv(os.path.join(os.path.dirname(__file__), prices_file), index_col=0, parse_dates=True)
    
    print(prices.loc[prices.index[1:100]]['AMZN'])

    plt.plot(prices.loc[prices.index[50:150]]['AMZN'])
    plt.savefig('stock_price.png')
    return

def stockpricefig():
    prices_file = 'SFFRL_data/price.csv' #'prices.csv

    prices = pd.read_csv(os.path.join(os.path.dirname(__file__), prices_file), index_col=0, parse_dates=True)#[591:791]
    d0 = prices.loc[prices.index[0]].copy()


    for i in prices.index:
         prices.loc[i] = prices.loc[i]/d0
    print(prices)
    prices.plot()
    plt.savefig('price-10.png')
    plt.tight_layout()
    return


def SFFRLvsBaselinesFig():
    SFF = pd.read_csv("PV/SFFRL.csv", index_col=0, parse_dates=True)
    OLMAR = pd.read_csv('PV/OLMAR.csv', index_col=0, parse_dates=True)
    UCRP = pd.read_csv('PV/UCRP.csv', index_col=0, parse_dates=True)
    WMAMR = pd.read_csv('PV/WMAMR.csv', index_col=0, parse_dates=True)
    #CORN = pd.read_csv('CORN.csv', index_col=0, parse_dates=True)
    #DynamicCRP = pd.read_csv('DynamicCRP.csv', index_col=0, parse_dates=True)
    MPT = pd.read_csv('PV/MPT.csv', index_col=0, parse_dates=True)

    testX = OLMAR.index
    plt.figure(num=1, figsize=(9, 5))

    plt.plot(testX, SFF.loc[:]['SFFRL'], label = 'SFFRL', ls='-')
    plt.plot(OLMAR.loc[:]['OLMAR'], label = 'OLMAR', ls='--')
    plt.plot(UCRP.loc[:]['UCRP'], label = 'UCRP', ls='-.')
    plt.plot(WMAMR.loc[:]['WMAMR'], label = 'WMAMR', ls=':')
    plt.plot(MPT.loc[:]['MPT'], label = 'MPT', ls=(0, (3, 1, 1, 1, 1, 1)))
    plt.grid()
    plt.xlabel('date')
    plt.ylabel('portfolio value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('PV_plot/pv-10.pdf')

SFFRLvsBaselinesFig()
#SFFRLvsBaselinesFig()