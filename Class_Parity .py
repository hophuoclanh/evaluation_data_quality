import numpy as np

def Class_Parity(Ci):
    Ni = Ci.values
    ni = Ni / np.max(Ni)
    n_mean = np.mean(ni)
    QoD_CP = 1 - np.sum(np.abs(ni - n_mean)) / len(Ni)
    return QoD_CP

