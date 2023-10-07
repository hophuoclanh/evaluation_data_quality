import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import TomekLinks
import cle

data = pd.read_csv('DATA.csv')
df = data.drop(columns='STUDENT ID')
X = df.drop(columns='GRADE')
y = df['GRADE']
def Class_Overlap(x,y):
    tl = TomekLinks()
    x_res, y_res = tl.fit_resample(x, y)
    di = x.iloc[0]
    total_data_samples = len(x_res)
    DoQ_CO = 1 - (len(di) / total_data_samples)
    return DoQ_CO

DoQ_CO =Class_Overlap(X, y)
print("Overlap Regions (QoD_CD):", DoQ_CO)

def Class_Parity(Ci):
    Ni = Ci.values
    ni = Ni / np.max(Ni)
    n_mean = np.mean(ni)
    QoD_CP = 1 - np.sum(np.abs(ni - n_mean)) / len(Ni)

    return QoD_CP
QoD_CP =   Class_Parity(y)
print('Class Parity =', QoD_CP)

def