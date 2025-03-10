from spinup import ppo_pytorch as ppo
from portfolio_env import PortfolioEnv
import gym
import torch


#env_fn = lambda: gym.make('gym_portfolio:portfolio-v0')
env_fn = PortfolioEnv

ac_kwargs = dict(hidden_sizes=[64, 64, 64], activation=torch.nn.Tanh)

logger_kwargs = dict(output_dir='output', exp_name='experiment_name')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=300, max_ep_len=10000, logger_kwargs=logger_kwargs)