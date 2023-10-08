import numpy as np
from imblearn.under_sampling import TomekLinks

def Class_Overlap(x,y):
    tl = TomekLinks()
    x_res, y_res = tl.fit_resample(x, y)
    di = x.iloc[0]
    total_data_samples = len(x_res)
    DoQ_CO = 1 - (len(di) / total_data_samples)
    return DoQ_CO

def Class_Parity(Ci):
    Ni = Ci.values
    ni = Ni / np.max(Ni)
    n_mean = np.mean(ni)
    QoD_CP = 1 - np.sum(np.abs(ni - n_mean)) / len(Ni)

    return QoD_CP

# example DATA https://www.kaggle.com/datasets/joebeachcapital/students-performance?fbclid=IwAR2ZJZALZrT0wupHkhc3vQZwtAwajcTkbRSFla3mt92_ggYPS0YGNUjldoE
import pandas as pd

data = pd.read_csv('DATA.csv')
df = data.drop(columns='STUDENT ID')

# select data example 'GRADE'
X = df.drop(columns='GRADE')
y = df['GRADE']

DoQ_CO =Class_Overlap(X, y)
print("Overlap Regions (QoD_CD):", DoQ_CO)

QoD_CP =   Class_Parity(y)
print('Class Parity =', QoD_CP)