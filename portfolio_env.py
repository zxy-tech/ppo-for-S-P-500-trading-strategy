""" Modified from https://github.com/awslabs/amazon-sagemaker-examples """

import gym
# import gym.spaces
import os
import numpy as np
import pandas as pd
import copy
import yaml
# from gym.utils import seeding

EPS = 1e-8


class PortfolioEnv(gym.Env):
    """ This class creates the financial market environment that the Agent interact with.

    It extends the OpenAI Gym environment https://gym.openai.com/.

    More information of how it is integrated into AWS is found here
    https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-rl-environments.html#sagemaker-rl-environments-gym

    The observations include a history of the signals with the given `window_length` ending at the current date.

    Args:
        steps (int):  Steps or days in an episode.
        trading_cost (float):  Cost of trade as a fraction.
        window_length (int):  How many past observations to return.
        start_date_index (int | None):  The date index in the signals and price arrays.

    Attributes:
        action_space (gym.spaces.Box):  [n_tickers]  The portfolio weighting not including cash.
        dates (np.array of np.datetime64):  [n_days] Dates for the signals and price history arrays.
        info_list (list):  List of info dictionaries for each step.
        n_signals (int):  Number of signals in each observation.
        n_tickers (int):  Number of tickers in the price history.
        observation_space (gym.spaces.Box)  [self.n_signals, window_length]  The signals with a window_length history.
        portfolio_value (float):  The portfolio value, starting with $1 in cash.
        gain (np.array):  [n_days, n_tickers, gain] The relative price vector; today's / yesterday's price.
        signals (np.array):  [n_signals, n_days, 1]  Signals that define the observable environment.
        start_date_index (int):  The date index in the signals and price arrays.
        step_number (int):  The step number of the episode.
        steps (int):  Steps or days in an episode.
        tickers (list of str):  The stock tickers.
        trading_cost (float):  Cost of trade as a fraction.
        window_length (int):  How many past observations to return.
    """

    def __init__(self, steps=500, trading_cost=0.0025, window_length=1, start_date_index=91):
        """An environment for financial portfolio management."""
        with open('settings.yml') as f:
            settings = yaml.safe_load(f)

        # Initialize some local parameters
        self.csv = 'output/portfolio-management.csv'
        self.weights_csv = 'output/weights.csv'
        self.info_list = list()
        self.weights_list = list()
        self.portfolio_value = 1.0
        self.step_number = 0
        self.prices_file = settings['prices_file'] #'sffprices.csv
        self.signals_file = settings['signals_file'] #sffobs.csv
        self.gain_file = settings['gain_file'] #'sffgain.csv'
        # Save some arguments as attributes
        self.trading_cost = trading_cost
        self.window_length = window_length

        # Read the stock data and convert to the relative price vector (gain)
        #   Note the raw prices have an extra day vs the signals to calculate gain
        #raw_prices = pd.read_csv(os.path.join(os.path.dirname(__file__), self.prices_file), index_col=0, parse_dates=True)
        #self.tickers = raw_prices.columns.tolist()
        self.gain = pd.read_csv(os.path.join(os.path.dirname(__file__), self.gain_file), index_col=0, parse_dates=True)
        self.tickers = self.gain.columns.tolist()
        self.dates = self.gain.index
        self.gain = self.gain.values
        self.n_dates = self.dates.shape[0]
        self.n_tickers = len(self.tickers)
        self.weights = np.insert(np.zeros(self.n_tickers), 0, 1.0)

        # Read the signals
        self.signals = pd.read_csv(os.path.join(os.path.dirname(__file__), self.signals_file),
                                   index_col=0, parse_dates=True).T.values[:, :, np.newaxis]
        self.n_signals = self.signals.shape[0]

        # Define the action space as the portfolio weights where wn are [0, 1] for each asset not including cash
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_tickers,), dtype=np.float32)

        # Define the observation space, which are the signals
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.n_signals * self.window_length,),
                                                dtype=np.float32)
        # Rest the environment
        self.start_date_index = start_date_index
        self.steps = steps
        self.reset()
            # -----------------------------------------------------------------------------------

    def step(self, action):
        """Step the environment.

        See https://gym.openai.com/docs/#observations for detailed description of return values.

        Args:
            action (np.array):  The desired portfolio weights [w0...].

        Returns:
            np.array:  [n_signals, window_length, 1] The observation of the environment (state)
            float:  The reward received from the previous action.
            bool:  Indicates if the simulation is complete.
            dict:  Debugging information.
        """
        step_interval = 1
        self.step_number += step_interval

        # Force the new weights (w1) to (0.0, 1.0) and sum weights = 1, note 1st weight is cash.
        w1 = np.clip(action, a_min=0, a_max=1)
        w1 = np.insert(w1, 0, np.clip(1 - w1.sum(), a_min=0, a_max=1))
        w1 = w1 / w1.sum()

        # Calculate the reward; Numbered equations are from https://arxiv.org/abs/1706.10059
        t = self.start_date_index + self.step_number
        y1 = self.gain[t]
        y1 = np.insert(y1,0,1.0)
        w0 = self.weights
        p0 = self.portfolio_value
        dw1 = (y1 * w0) / (np.dot(y1, w0) + EPS)            # (eq7) weights evolve into
        mu1 = self.trading_cost * (np.abs(dw1 - w1)).sum()  # (eq16) cost to change portfolio
        p1 = p0 * (1 - mu1) * np.dot(y1, w1)                # (eq11) final portfolio value
        p1 = np.clip(p1, 0, np.inf)                         # Limit portfolio to zero (busted)
        rho1 = p1 / p0 - 1                                  # rate of returns
        reward = np.log((p1 + EPS) / (p0 + EPS))            # log rate of return

        # Save weights and portfolio value for next iteration
        self.weights = w1
        self.portfolio_value = p1

        # Observe the new environment (state)
        t0 = t - self.window_length + 1
        observation = self.signals[:, t0:t+1, :]
        observation = np.reshape(observation, (-1))
        # Save some information for debugging and plotting at the end
        r = y1.mean()
        if self.step_number == step_interval:
            market_value = r
        else:
            market_value = self.info_list[-1]["market_value"] * r 
        info = {"reward": reward, "log_return": reward, "portfolio_value": p1, "return": r, "rate_of_return": rho1,
                "weights_mean": w1.mean(), "weights_std": w1.std(), "cost": mu1, 'date': self.dates[t],
                'steps': self.step_number, "market_value": market_value}
        self.info_list.append(info)
        
        #weight_info = copy.deepcopy(self.weights)
        weight_info = list(self.weights)
        weight_info.insert(0, self.dates[t])
        self.weights_list.append(weight_info)

        # Check if finished and write to file
        done = False
        if (self.step_number >= self.steps) or (p1 <= 0):
            done = True
            pd.DataFrame(self.info_list).sort_values(by=['date']).to_csv(self.csv)
            weights_list_columns = copy.deepcopy(self.tickers)
            weights_list_columns.insert(0, "CASH")
            weights_list_columns.insert(0, "date")
            pd.DataFrame(self.weights_list, columns=weights_list_columns).sort_values(by=['date']).to_csv(self.weights_csv)
        return observation, reward, done, info

    def reset(self):
        """Reset the environment to the initial state.

        Ref:
        https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-rl-environments.html#sagemaker-rl-environments-gym
        """
        self.info_list = list()
        self.weights_list = list()
        self.weights = np.insert(np.zeros(self.n_tickers), 0, 1.0)
        self.portfolio_value = 1.0
        self.step_number = 0

        # Limit the number of steps
        self.steps = min(self.steps, self.n_dates - self.window_length - 1)

        # Control the start date
        if self.start_date_index is None:
            self.start_date_index = np.random.random_integers(self.window_length - 1, 
                                                              self.n_dates - self.steps - 1)
        else:     
            # noinspection PyTypeChecker
            self.start_date_index = np.clip(self.start_date_index,
                                            a_min=self.window_length - 1,
                                            a_max=self.n_dates - self.steps - 1)

        t = self.start_date_index + self.step_number
        t0 = t - self.window_length + 1
        observation = self.signals[:, t0:t+1, :]
        observation = np.reshape(observation, (-1))

        return observation
