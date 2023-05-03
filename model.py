import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import RidgeCV, LassoCV, Lasso
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import sys
 
n = int(sys.argv[1])
data = pd.read_csv('traindata.csv')
data = data.drop(['id','price','date','grade' ,'price/sqft', 'rootprice/sqft', 'sqft_living15', 'sqft_lot15'], axis=1)
X_train = data.drop(['log price'], axis=1)
y_train = data['log price']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("Finding best parameters for Gradient Boosting Regressor...")

model_params = GradientBoostingRegressor()
param_grid = {
    'n_estimators': [60, 80, 100, 120, 140],
    'learning_rate': [0.4, 0.2, 0.1, 0.05],
    'max_depth': [5, 6, 7, 8, 9, 10]
}

grid_search = GridSearchCV(estimator=model_params, param_grid=param_grid, cv=10)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
if n == 1:
    y_pred = grid_search.best_estimator_.predict(X_test)
    rmse_gradient = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE on test data: ", rmse_gradient) 
    print("R^2 on test data: ", grid_search.best_estimator_.score(X_test, y_test))
    print("R^2 on train data: ", grid_search.best_estimator_.score(X_train, y_train))


X_train_importance = X_train[['bedrooms','bathrooms','sqft_living','view', 'yr_built','yr_renovated','zipcode','lat','long','age', 'in_expensive_zip']]
X_test_importance = X_test[['bedrooms','bathrooms','sqft_living','view', 'yr_built','yr_renovated','zipcode','lat','long','age', 'in_expensive_zip']]

grid_search_importance = GridSearchCV(estimator=model_params,
                        param_grid={key: [value] for key, value in grid_search.best_params_.items()}, cv=10)
grid_search_importance.fit(X_train_importance, y_train)
print("Parameters used with important features: ", grid_search.best_params_)
print("Score with important features: ", grid_search.best_score_)

if n == 1:
    y_pred = grid_search_importance.best_estimator_.predict(X_test_importance)
    rmse_gradient = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE on test data with important features: ", rmse_gradient) 
    print("R^2 on test data with important features: ", grid_search_importance.best_estimator_.score(X_test_importance, y_test))
    print("R^2 on train data with important features: ", grid_search_importance.best_estimator_.score(X_train_importance, y_train))

