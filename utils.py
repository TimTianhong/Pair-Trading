import pandas as pd
import numpy as np

def corr(y1, y2):
    return pd.DataFrame({'y1': np.array(y1), 'y2': np.array(y2)}).corr().iloc[0, 1]

def top_n_elements(d, n, reverse=True):
    sorted_dict = dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse)[:n])
    return sorted_dict