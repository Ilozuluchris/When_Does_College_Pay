import numpy as np
import pandas as pd

region_data =  pd.read_csv('salaries-by-region.csv').dropna().drop_duplicates()