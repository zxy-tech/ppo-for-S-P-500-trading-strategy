import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

progress = pd.read_csv("output/progress.txt", sep='\t')
AvgEpRet = progress['AverageEpRet'].to_numpy()
MaxEpRet = progress['MaxEpRet'].to_numpy()
MinEpRet = progress['MinEpRet'].to_numpy()
x = np.array(range(AvgEpRet.shape[0]))

plt.figure(num=1, figsize=(9, 5))
plt.plot(x, AvgEpRet)
plt.fill_between(x, MinEpRet, MaxEpRet, alpha=0.4)
plt.grid()
plt.xlabel('epochs')
plt.ylabel('cummulative return')
plt.tight_layout()
plt.savefig("AverageEpRet.pdf")

AverageVVals = progress['AverageVVals'].to_numpy()
MaxVVals = progress['MaxVVals'].to_numpy()
MinVVals = progress['MinVVals'].to_numpy()
plt.figure(num=2, figsize=(9, 5))
plt.plot(x, AverageVVals)
plt.fill_between(x, MinVVals, MaxVVals, alpha=0.4)
plt.grid()
plt.xlabel('epochs')
plt.ylabel('AverageVVals')
plt.tight_layout()
plt.savefig("AverageVVals.pdf")
