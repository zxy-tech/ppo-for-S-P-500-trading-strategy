import numpy as np
import pandas as pd

def calMDD():
    files = ['SFFRL', 'UCRP', 'OLMAR', 'WMAMR', 'MPT']
    for file in files:
        df = pd.read_csv("PV/"+file+".csv", index_col=0, parse_dates=True)[file]
        MDD = 0
        peak = -99999
        DD = []
        for i in range(df.shape[0]):
            if df.iloc[i] > peak:
                peak = df.iloc[i]
            DD = np.append(DD, (peak - df.iloc[i])/peak)
            if DD[i] > MDD:
                MDD = DD[i]
        print(file, ": ", MDD)

calMDD()