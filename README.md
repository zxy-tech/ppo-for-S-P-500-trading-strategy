# PPO-for-S-P-500-trading-strategy

++++++++++++++++++++++++++++++++++++++++++++++++++++++  
20250310 by user Xiaoyan Zhang, Lingnan College, SYSU:  
++++++++++++++++++++++++++++++++++++++++++++++++++++++  

--------------------------------------------------------------------------------------------  
--------------------------------------------------------------------------------------------  

该文档基于 Shi Runhao 在 2021 年的本科毕业论文所编辑而成，并对其中的代码进行了一定程度的维护和修改（特别是针对 Mac 操作系统的修改）。

- **数据**：dataset 手动收集自 Yahoo Finance 数据接口 yfinance 中 S&P 500 所有成分股的收盘价（复权后），自 2013 年至 2025 年 3 月。有部分数据缺失，由于成分股存在变动问题，因此产生缺失。

- **运行环境**：spinup（spinningup）是 OpenAI 在 2017 年开发的针对 PPO、GRPO 等强化学习策略所开发的框架。但是由于兼容问题只能在较低版本的 Python 上运行，因此需要主动降低 Python 版本环境至一致后才能运行。对于 Mac 版本用户（arm64 架构），有些包是没办法使用的，因此必须按照上述方法手动更改包的设置才能运行。

  除此之外，Runhao 学姐整合了一个新的环境，用于股票交易问题的重述，具体在 `portfolio_eny.py` 中。这个类为强化学习提供了一个金融市场环境，智能体可以在这个环境中进行交易操作。通过继承 OpenAI Gym 的环境类，它可以使用标准的 Gym 接口来定义动作空间、观察空间等。此外，它还支持与 AWS SageMaker 的集成，便于在云环境中进行强化学习的训练和部署。

- **运行顺序**：
  - 通过 `sff_part` 先生成 `sff_gain`、`sff_signal` 和 `sff_price` 三个文件，对应于 `settings.yaml`。
  - 通过 `sff-arima` 生成 `sff-arima` 预测和拟合图和结果。
  - 通过 `baseline` 生成不同基础策略的交易结果和数据。
  - 通过 `drl-portfolio-main` 进行深度学习训练。
  - 通过 `test_sffrl` 生成 PPO 交易结果和数据。
  - 通过 `progress_plot` 生成 PPO 的训练效果。
  - 通过 `sharpe` 和 `mdd` 生成夏普率和最大回撤率。
  - 通过 `thesis_plot` 生成不同 benchmarks 和模型的交易结果并画图。

--------------------------------------------------------------------------------------------  
--------------------------------------------------------------------------------------------  

## 环境安装

```bash
pip command: pip install -r requirements.txt
conda command: conda env create -f environment.yml
```

## For Mac Users
由于 Mac 用户无法访问较低版本的 Python（<=3.9），因此需要访问 issue 文件 #381 以成功部署 spinningup：[https://github.com/openai/spinningup/pull/381](https://github.com/openai/spinningup/pull/381)
```bash
git clone https://github.com/openai/spinningup.git
cd spinningup
git fetch origin pull/381/head:2023Jan_dependency_upgrades
git checkout 2023Jan_dependency_upgrades
pipenv --python 3.9
pipenv shell
pip install swig
brew install freetype
pip install pyglet==1.5.15
pip install tensorflow-macos
pip install tensorflow-metal
```
替换 setup.py 中的 tensorflow>=1.8.0,<3.0 为：
```bash
'tensorflow>=1.8.0,<3.0; sys_platform != "darwin" or platform_machine != "arm64"',
'tensorflow-macos>=1.8.0,<3.0; sys_platform == "darwin" and platform_machine == "arm64"',
```
最后运行安装命令：
```bash
pip install -e .
```

注意：如果使用该设置，需要额外更改 spinningup setup.py 中的 matplotlib==3.5.0rc1，并在手动安装成功后卸载并安装稳定版本：
```bash
pip install matplotlib==3.3.4
```
手动安装方式为：
```bash
gh repo clone openai/spinningup | or pip install https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
```
