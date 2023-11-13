from TomekLinks import  tomek_links

def Class_Overlap(X,y):
    tl = tomek_links(X,y)
    x_res, y_res = tl.fit_resample(X, y)
    di = X.iloc[0]
    total_data_samples = len(x_res)
    DoQ_CO = 1 - (len(di) / total_data_samples)
    return DoQ_CO