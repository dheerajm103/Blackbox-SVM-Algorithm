import pandas as pd                          # importing library 
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("forestfires.csv")          # importing dataset

# data cleansing and eda part ***************************************************************************
df.info()                                    # checking for data types and null values
df.describe()                                # checking for mean ,median and sd                       
df.duplicated().sum()                        # checking for duplicate records
df.drop_duplicates()
df=df.drop(["month","day"], axis = 1)        # dummy column for categorical column  
df.nunique()
plt.boxplot(df)                              # checking outliers
df.corr()                                    # checking correlation
df.skew()                                    # checking skewness
df.kurtosis()                                # checking kurtosis
df = pd.get_dummies(df , drop_first = True)

def norm1(i):                                # normalization
	x = (i-i.min())	/(i.max()-i.min())
	return(x)

norm = norm1(df)


# model building****************************************************************************************

train,test = train_test_split(df, test_size = 0.20)

train_X = train.iloc[:, 0:28]
train_y = train.iloc[:, 28]
test_X  = test.iloc[:, 0:28]
test_y  = test.iloc[:, 28]


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
