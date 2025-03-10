import gym
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import seaborn as sns
sns.set_context("notebook")
plt.rcParams["figure.figsize"] = (18, 10)

from universal.algo import Algo
import universal as up

# ignore logged warnings
import logging
logging.getLogger().setLevel(logging.ERROR)

class SFFRL(Algo):
    def __init__(self):
        with open('settings.yml') as f:
            settings = yaml.safe_load(f)
        PATH = settings['model_path']
        self.model = torch.load(PATH)
        self.model.eval()
        signals_file = settings['signals_file'] #signals.csv
        self.signals = pd.read_csv(os.path.join(os.path.dirname(__file__), signals_file), index_col=0, parse_dates=True)[591:791]
        self.signals = self.signals.apply(pd.to_numeric, errors='coerce')
        self.date_index = self.signals.index
        self.idx = 0
        super(SFFRL, self).__init__()

    def init_weights(self, cols):
        return np.insert(np.zeros(len(cols)-1), 0, 1.0)

    def step(self, x, last_b, history):
        # calculate moving average
        cur_date = self.date_index[self.idx]
        observation = self.signals.loc[cur_date].values
        observation = np.reshape(observation, (-1))
        observation = torch.tensor(observation, dtype=torch.float)
        action = self.model.act(observation)
        weight = np.clip(action, a_min=0, a_max=1)
        weight = np.insert(weight, 0, np.clip(1 - weight.sum(), a_min=0, a_max=1))
        weight = weight / weight.sum()
        # normalize so that they sum to 1
        return weight

with open('settings.yml') as f:
    settings = yaml.safe_load(f)
prices_file = settings['prices_file'] #'prices.csv
data = pd.read_csv(prices_file, index_col=0, parse_dates=True)#LR:[200:900] MR:[700:900] SR:[850,900] AT:[0:900] 
data = data.apply(pd.to_numeric, errors='coerce')

d0 = data.loc[data.index[0]].copy()
data.insert(0, "CASH", np.ones(data.shape[0]))

sffrl = SFFRL()
result = sffrl.run(data)
result.fee = 0.0025
print(result.summary())

ax1, ax2 = result.plot()
plt.clf()

result.plot(weights=False, assets=False, ucrp=False, logy=True)
ax = plt.gca()
line = ax.lines[0]
print(line.get_ydata())
df = pd.DataFrame(line.get_ydata(), columns=['SFFRL'], index=data.index)
df.to_csv("PV/SFFRL.csv")
plt.tight_layout()
plt.savefig("baselines_plot/SFFRL.pdf")
plt.clf()

result.plot_decomposition(legend=True, logy=True)
plt.tight_layout()
plt.savefig("baselines_plot/SFFRL_decomposition.png")
plt.clf()

result.plot()
plt.tight_layout()
plt.savefig("baselines_plot/SFFRL_weights.png")
