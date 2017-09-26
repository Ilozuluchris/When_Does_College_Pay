import numpy as np
import pandas as pd

region_data =  pd.read_csv('salaries-by-region.csv',keep_default_na=False)

region_data_array = np.array(region_data)

print region_data.head()