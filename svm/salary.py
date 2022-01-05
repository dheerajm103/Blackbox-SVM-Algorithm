import pandas as pd                                            # importing library 
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df_train = pd.read_csv("SalaryData_Train.csv")                 # importing dataset
df_test = pd.read_csv("SalaryData_Test.csv")

# data cleansing and eda part ***************************************************************************
df_train.info()                                                # checking for data types and null values
df_test.info() 
df_train.describe()                                            # checking for mean ,median and sd                       
df_test.describe() 
df_train.duplicated().sum()                                    # checking for duplicate records
df_test.duplicated().sum()
df_train.drop_duplicates()
df_test.drop_duplicates()
df_train=df_train.drop(["educationno"], axis = 1)              # dropping nominal column  
df_test=df_test.drop(["educationno"], axis = 1)
df_train.nunique()
df_train = pd.get_dummies(df_train , drop_first = True)        # dummy column for categorical column  
df_test = pd.get_dummies(df_test , drop_first = True)
plt.boxplot(df_train)                                          # checking outliers
plt.boxplot(df_test)
df_train.corr()                                                # checking correlation
df_train.skew()                                                # checking skewness
df_train.kurtosis()                                            # checking kurtosis

def norm1(i):                                                  # normalization
	x = (i-i.min())	/(i.max()-i.min())
	return(x)

norm_train = norm1(df_train)
norm_test = norm1(df_test)

# model building****************************************************************************************

train_X = norm_train.iloc[:, 0:93]
train_y = norm_train.iloc[:, 93]
test_X  = norm_test.iloc[:, 0:93]
test_y  = norm_test.iloc[:, 93]


# kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)

pred_test_linear = model_linear.predict(test_X)     # test accuracy
np.mean(pred_test_linear == test_y)

pred_test_linear = model_linear.predict(train_X)    # train accuracy
np.mean(pred_test_linear == train_y)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)

pred_test_rbf = model_rbf.predict(test_X)           # test accuracy
np.mean(pred_test_rbf == test_y)

pred_train_rbf = model_linear.predict(train_X)      # train accuracy
np.mean(pred_train_rbf == train_y)
