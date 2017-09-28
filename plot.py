import numpy as np
import pandas as pd

def import_and_clean_csv(csv):
    """
    :param csv: string describing path to csv to be imported
    :return: pandas data frame containing clean data.
    """
    return pd.read_csv(csv).dropna().drop_duplicates()


region_data = import_and_clean_csv('salaries-by-region.csv')

print region_data.head()