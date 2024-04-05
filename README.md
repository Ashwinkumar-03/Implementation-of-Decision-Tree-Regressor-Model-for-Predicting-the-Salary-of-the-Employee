# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas. 

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset. 

4.calculate Mean square error,data prediction and r2.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Ashwin Kumar S
RegisterNumber:  212222240013
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```


## Output:
### Initial dataset:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118663725/d596761a-52f9-49a3-80d3-c694f398acb3)

### Data Info:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118663725/be6df8ef-e44b-48c3-a20a-4cceddcdd6c1)

### Optimization of null values:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118663725/256e9979-f1f7-4d1f-9cd9-f0d03962685f)

### Converting string literals to numerical values using label encoder:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118663725/3e99bc9b-c977-4b90-90a3-387babce302d)

### Assigning x and y values:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118663725/366db3ed-2ac3-4d4a-b9c1-8d3c7f28ab66)

### Mean Squared Error:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118663725/7233a68e-8ff3-42da-8bd8-43fd0585dd99)

### R2 (variance):
![image](https://github.com/Ashwinkumar-03/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118663725/060b927a-7d1a-48fa-832d-5cc4bf47711b)

### Prediction:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118663725/afa60a68-5464-4e7d-bd5b-d84a985603de)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
