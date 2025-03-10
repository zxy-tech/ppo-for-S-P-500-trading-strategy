# ppo-for-S-P-500-trading-strategy
++++++++++++++++++++++++++++++++++++++++++++++++++++++
20250310 by user Xiaoyan Zhang, Lingnan College, SYSU:
++++++++++++++++++++++++++++++++++++++++++++++++++++++

--------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------

该文档基于 Shi Runhao在2021年的本科毕业论文所编辑而成，并对其中的代码进行了一定程度的维护和修改（特别是针对mac操作系统的修改）。

-数据：dataset 手动收集字yahoofiance 数据接口 yfinance 中S&P 500 所有成分股的收盘价（复权后） 自2013年至2025年3月 有部分数据缺失，由于成分股存在变动问题，因此产生缺失

-运行环境： spinup（spinningup）是openai在2017年开发的针对 PPO，GRPO 等强化学习策略所开发的框架，但是由于兼容问题只能在较低版本的python上运行，因此需要主动降低python版本环境至一致后才能运行，但是对于mac版本用户（arm64架构），有些包是没办法使用的因此必须按照上述方法手动更改包的设置才能运行
          除此之外，Runhao 学姐整合了一个新的环境，用于股票交易问题的重述具体在 portfolio_eny.py中,这个类为强化学习提供了一个金融市场环境，智能体可以在这个环境中进行交易操作。通过继承OpenAI Gym的环境类，它可以使用标准的Gym接口来定义动作空间、观察空间等。此外，它还支持与AWS SageMaker的集成，便于在云环境中进行强化学习的训练和部署。
-运行顺序：    
            ---通过sff_part先生成 'sff_gain'，'sff_signal', 和 'sff_price'三个文件，对应于 settings.yaml
            ---通过sff-arima生成sff-arima预测和拟合图和结果
            ---通过baseline生成不同基础策略的交易结果和数据
            ---通过drl-portfolio-main进行深度学习训练
            ---通过test_sffrl生成ppo交易结果和数据
            ---通过progress_plot生成ppo的训练效果
            ---通过sharpe和mdd生成夏普率和最大回撤率
            ---通过sharpe和mdd生成夏普率和最大回撤率
            ---通过thesis_plot生成不同benchmarks和模型的交易结果并画图
--------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------


环境安装:
pip command: pip install -r requirements.txt
conda command: conda env create -f environment.yml


--------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------
For mac users, due to the fact that you cannot access python' lower verison (<=3.9) , you are necessary to access the issue issue file #381 for the following command to secessfully deploy spinningup  [https://github.com/openai/spinningup/pull/381] 
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

replace 'tensorflow>=1.8.0,<3.0 in setup.py with:
'tensorflow>=1.8.0,<3.0; sys_platform != "darwin" or platform_machine != "arm64"',
'tensorflow-macos>=1.8.0,<3.0; sys_platform == "darwin" and platform_machine == "arm64"',

Finally run the install:
pip install -e .

PS: 注意如果使用改设置需要额外更改spinningup setup.py 中为 'matplotlib==3.5.0rc1' 且在手动安装成功后卸载在安装稳定版本 pip install matplotlib==3.3.4
手动安装方式为:
gh repo clone openai/spinningup | or pip install https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
--------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------




https://www.reddit.com/r/reinforcementlearning/comments/c2zog4/state_transition_probability_and_policy_difference/



