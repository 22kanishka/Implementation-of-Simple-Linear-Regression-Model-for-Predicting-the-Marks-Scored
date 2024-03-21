# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: kanishka p
RegisterNumber:  2305001011

```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![image](https://github.com/22kanishka/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145959493/f0f41261-ad19-48a6-b8b9-7fd20f859b93)

![image](https://github.com/22kanishka/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145959493/a241b55f-33d4-407c-a30d-ff65b3940b43)

![image](https://github.com/22kanishka/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145959493/680c5708-5dcf-4621-bbe1-0256f10b1b49)

![image](https://github.com/22kanishka/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145959493/a55c8717-e0c4-4a39-8710-0d2149bdd897)

![image](https://github.com/22kanishka/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145959493/93d6aa72-c44e-4ec2-a33b-5b10fb07bd0f)

![image](https://github.com/22kanishka/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145959493/2886fb49-7095-4f0d-a68b-8cfd21032901)

![image](https://github.com/22kanishka/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145959493/0f5c9d4d-3e1b-41e6-9fe8-1c00df66d6bf)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
