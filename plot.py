import numpy as np
import pandas as pd

region_data =  pd.read_csv('salaries-by-region.csv').dropna()

print region_data.head()

print region_data.drop_duplicates()