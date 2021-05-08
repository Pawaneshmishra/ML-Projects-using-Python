#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Here we are fetching our data for training from a csv file.
path = r"https://drive.google.com/uc?export=download&id=1QSj8BMTUniLSQUQZnatKlfhARkYvWwNg"
df = pd.read_csv(path)

#Basic info of the data
df.describe()

#graph of data set
plt.scatter(x = df.time, y = df.mark, color = 'r')
plt.xlabel("Student Study Hours")
plt.ylabel("Students Marks")
plt.title("plot of Hours v/s Marks")
plt.show()

#checking for any invalid values in dataset
df.isnull().sum()

#data is clean
#cleaning is needed before starting
#assume that we had some empty or NULL values in table
#x = df.mean()
#df2=df.fillna(x)
#df2.isnull().sum() -> now this will give 0 which means data is clean.
#now if data is cleaned then in place of df use df2 everywhere.

#creating axes for our data.
X=df.drop("mark",axis="columns")
y=df.drop("time",axis="columns")
print("Shape of X is ",X.shape)
print("Shape of y is ",y.shape)

#dividing the dataset for tarining and testing
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 51)
print("Shape of X_train is ",X_train.shape)
print("Shape of X_test is ",X_test.shape)
print("Shape of y_train is ",y_train.shape)
print("Shape of y_test is ",y_test.shape)

#Linear Regression : y = m*x + c
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

#fitting our graph using linear regression
lr.fit(X_train,y_train)

#this is 'm' value
lr.coef_

#this is "c" value
lr.intercept_


#using [0][0] gives exact value rather than an array
lr.predict([[4]])[0][0].round(2)

pred_y = lr.predict(X_test)

#now we are changing the appearance of our data
pd.DataFrame(np.c_[X_test , y_test , pred_y] , columns = ["Study_Hours","Original_marks","Predicted_marks"])

#shows accuracy of training
lr.score(X_test,y_test)

#training set visualisation
plt.scatter(X_train,y_train, color = 'g')

plt.scatter(X_test,y_test)
plt.plot(X_train,lr.predict(X_train),color='r')

#importing our results into a single file
import joblib
joblib.dump(lr , "Student_Marks_Prediction_ML_Model.pkl")

#creating an object of our model
model=joblib.load("Student_Marks_Prediction_ML_Model.pkl")

#code for basic user interaction!!!

x1=int(input("Enter the amount of hours you study :"))
y1=model.predict([[x1]])[0][0].round(2)
if y1>=100:
  print("If you study {0} hours, you will get 100% marks!!".format(x1))
else:
  print("If you study {0} hours, you will get {1} marks!!".format(x1,y1))
