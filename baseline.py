# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'svg'

import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
import yaml

import universal as up
from universal import algos
from universal.algos import *

sns.set_context("notebook")
plt.rcParams["figure.figsize"] = (16, 8)

# ignore logged warnings
import logging
logging.getLogger().setLevel(logging.ERROR)

def OLMAR(data, savefile=False, savefig=False):
    algo = algos.OLMAR(window=5, eps=10)
    result = algo.run(data)
    result.fee = 0.0025
    print(result.summary())
    result.plot(weights=False, assets=False, logy=True)

    ax = plt.gca()
    line = ax.lines[0]
    print(line.get_ydata())
    if savefile:
        df = pd.DataFrame(line.get_ydata(), columns = ['OLMAR'], index = data.index)
        df.to_csv("PV/OLMAR.csv")
    if savefig==True:
        plt.tight_layout()
        plt.savefig("baselines_plot/OLMAR.png")
        plt.clf()
        
        plt.tight_layout()
        result.plot_decomposition(legend=True, logy=True)
        plt.savefig("baselines_plot/OLMAR_decomposition.png")
        plt.clf()

        plt.tight_layout()
        result.plot()
        plt.savefig("baselines_plot/OLMAR_weights.png")

    return result


def UCRP(data, savefile=False, savefig=False):
    algo = algos.CRP()
    result = algo.run(data)
    result.fee = 0.0025
    print(result.summary())
    result.plot(weights=False, assets=False, logy=True)

    ax = plt.gca()
    line = ax.lines[0]
    print(line.get_ydata())
    if savefile:
        df = pd.DataFrame(line.get_ydata(), columns = ['UCRP'], index = data.index)
        df.to_csv("PV/UCRP.csv")

    if savefig==True:
        plt.tight_layout()
        plt.savefig("baselines_plot/UCRP.png")
        plt.clf()

        plt.tight_layout()
        result.plot_decomposition(legend=True, logy=True)
        plt.savefig("baselines_plot/UCRP_decomposition.png")
        plt.clf()
        plt.tight_layout()
        result.plot()
        plt.savefig("baselines_plot/UCRP_weights.png")
    
    return result

def WMAMR(data, savefile=False, savefig=True):
    algo = algos.WMAMR(window=5)
    result = algo.run(data)
    result.fee = 0.0025
    print(result.summary())
    result.plot(weights=False, assets=False, logy=True)

    ax = plt.gca()
    line = ax.lines[0]
    print(line.get_ydata())
    if savefile:
        df = pd.DataFrame(line.get_ydata(), columns = ['WMAMR'], index = data.index)
        df.to_csv("PV/WMAMR.csv")

    if savefig==True:
        plt.savefig("baselines_plot/WMAMR.png")
        plt.clf()

        result.plot_decomposition(legend=True, logy=True)
        plt.savefig("baselines_plot/WMAMR_decomposition.png")
        plt.clf()

        result.plot()
        plt.savefig("baselines_plot/WMAMR_weights.png")
    
    return result



def MPT(data, savefile=False, savefig=False):
    algo = algos.MPT(cov_window=52, min_history=4, cov_estimator='oas', method='mpt', q=0)
    result = algo.run(data) 
    result.fee = 0.0025
    print(result.summary())
    result.plot(weights=False, assets=False, ucrp=True, logy=True)

    ax = plt.gca()
    line = ax.lines[0]
    print(line.get_ydata())
    if savefile:
        df = pd.DataFrame(line.get_ydata(), columns = ['MPT'], index = data.index)
        df.to_csv("PV/MPT.csv")

    if savefig==True:
        plt.tight_layout()
        plt.savefig("baselines_plot/MPT.png")
        plt.clf()

        plt.tight_layout()
        result.plot_decomposition(legend=True, logy=True)
        plt.savefig("baselines_plot/MPT_decomposition.png")
        plt.clf()

        plt.tight_layout()
        result.plot()
        plt.savefig("baselines_plot/MPT_weights.png")

    return result

with open('settings.yml') as f:
    settings = yaml.safe_load(f)
prices_file = settings['prices_file'] #'prices.csv
data = pd.read_csv(prices_file, index_col=0, parse_dates=True)#LR:[200:900] MR:[700:900] SR:[850,900] AT:[0:900] 


# d0 = data.loc[data.index[0]].copy()


# for i in data.index:
#     data.loc[i] = data.loc[i]/d0

data.insert(0, "CASH", np.ones(data.shape[0]))

# data.plot()
# plt.legend()
# plt.savefig("data.png")

# OLMAR(data)
# UCRP(data)
# WMAMR(data)
# MPT(data)
OLMAR(data, savefile=True, savefig=True)
UCRP(data, savefile=True, savefig=True)
WMAMR(data, savefile=True, savefig=True)
MPT(data, savefile=True, savefig=True)

