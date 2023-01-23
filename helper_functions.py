import numpy as np
def price_change(values):
    change_array = []
    for i in range(len(values)-1):
        change = ((values[i] - values[i+1])/abs(values[i+1])) *100
        change_array.append(change)
    return np.array(change_array)