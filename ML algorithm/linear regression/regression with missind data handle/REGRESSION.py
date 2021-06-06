# Simple Linear Regression

# Importing the libraries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



# Importing the dataset
#dataset = pd.read_csv('Salary_Data.csv')
dataset = pd.read_csv('Blood_fat.csv')


#start
#dataset = pd.read_csv('Data.csv')

#missing darta handle
from sklearn.impute import SimpleImputer
#imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
imputer = SimpleImputer(missing_values = np.nan , strategy = 'constant',fill_value=30)
#imputer = SimpleImputer(missing_values = 'NaN',strategy = 'mean')
imputer = imputer.fit(dataset.iloc[:, :])
dataset.iloc[:, :] = imputer.transform(dataset.iloc[:, :])
print(dataset.iloc[:,:])

#colume allocated
X = pd.DataFrame(dataset.iloc[:, 0].values)
y = pd.DataFrame(dataset.iloc[:, 3].values)



#print(dataset)
#dataset.head(6)
#-1 in iloc indicates all columns except the last one
#change
#X = pd.DataFrame(dataset.iloc[:,1:8].values)
#y = pd.DataFrame(dataset.iloc[:, 1].values)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3.0, random_state = 0)

#whenever you execute your code a new random value is generated and the train and test datasets would have different values each time.
#However, if you use a particular value for random_state(random_state = 1 or any other value) everytime the result will be same,i.e, same values in train and test datasets.


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Blood fact (Training set)')
plt.xlabel('Index')
plt.ylabel('blood fat content')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
