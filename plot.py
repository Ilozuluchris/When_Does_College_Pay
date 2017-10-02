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
region_data.reshape()

#todo plot by school on  horizontal axis,vertical values , color by region,grid by levels of salary

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



#sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state=42)


# Import necessary modules
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import train_test_split



"""
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4,normalize=True)

# Fit the regressor to the data
lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()
"""
"""
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C':c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.6,random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg,param_grid,cv=5)

# Fit it to the training data
logreg_cv.fit(X_train,y_train)
# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))
"""

"""
# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import GridSearchCV,train_test_split
# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio':l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net,param_grid,cv=5)

# Fit it to the training data
gm_cv.fit(X_train,y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(y_test,y_pred)
mse = mean_squared_error(y_pred,y_test)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))
"""


"""
pipelines 
# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet',ElasticNet())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline,parameters,cv=3)

# Fit to the training set
gm_cv.fit(X_train,y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))

"""