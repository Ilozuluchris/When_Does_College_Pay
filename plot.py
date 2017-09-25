import numpy as np
import pandas as pd

region_data =  pd.read_csv('salaries-by-region.csv')


region_data_array = np.array(region_data)

for data in region_data_array:
    if data == n:
        data = 0
print region_data_array