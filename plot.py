import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
#todo plot by school on horizontal axis,vertical values , color by region,grid by levels of salary

#print region_data.info()

"""
School Name                          273 non-null object
Region                               273 non-null object
Starting Median Salary               273 non-null float64
Mid-Career Median Salary             273 non-null float64
Mid-Career 10th Percentile Salary    273 non-null float64
Mid-Career 25th Percentile Salary    273 non-null float64
Mid-Career 75th Percentile Salary    273 non-null float64
Mid-Career 90th Percentile Salary    273 non-null float64
dtypes: float64(6), object(2)
memory usage: 19.2+ KB
"""

#for plotting data
_ = sns.set();
fig = plt.figure(figsize=(20,20));
ax = fig.add_subplot(1,1,1)
ax2 = fig.add_subplot(2,1,1)
sns.swarmplot(x="Region", y="Starting Median Salary", data=region_data,ax=ax)
sns.boxplot(x="Region",  y="Starting Median Salary", data=region_data,ax=ax2)
plt.show()
#todo grid plots of box plot of region  vs all ranges of salary