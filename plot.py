import numpy as np
import pandas as pd

def import_and_clean_csv(csv):
    """
    :param csv: string describing path to csv to be imported
    :return: pandas data frame containing clean data.
    """
    return pd.read_csv(csv).dropna().drop_duplicates()


region_data = import_and_clean_csv('salaries-by-region.csv')

#print region_data.head() # to validate the csv was imported

cols = ["Starting Median Salary","Mid-Career Median Salary","Mid-Career 10th Percentile Salary","Mid-Career 25th Percentile Salary",
"Mid-Career 75th Percentile Salary",
"Mid-Career 90th Percentile Salary"]
region_data[cols] =region_data[cols].applymap(lambda x: pd.to_numeric(x.replace("$", "").replace(",", ""))) # removes $ sign and commas then converts to numeric values
'''
to validate above operation worked
print region_data.head()
print region_data.dtypes
'''