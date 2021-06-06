# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = pd.DataFrame(dataset.iloc[:, :-1].values)
y = pd.DataFrame(dataset.iloc[:, 3].values)
print(X[2])

# Taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis=0)
#0 mane column
#imputer = imputer.fit(X.iloc[:, 1:3])
#X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])


#Imputer was deprecated 3 versions ago and remove in 0.22

from sklearn.impute import SimpleImputer
#imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
imputer = SimpleImputer(missing_values = np.nan , strategy = 'constant',fill_value=30)
imputer = imputer.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])
print(X.iloc[:,1:3])
