#import pandas for data processing
import pandas as pd
#import numpy for matrices
import numpy as np
#import matplotlib for ploting graphs
import matplotlib.pyplot as plt
#import sklearn features used for calculating RMSE, fit regression model and Polynomial features
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#import seaborn for heatmaps
import seaborn as sns
#import sklearn features for score functions
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest  
from sklearn.feature_selection import f_classif 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils

#read xlsx file/create dataframe
df = pd.read_excel('data/datasets.xlsx')
#drop unnecessary fields from the dataframe
df = df.drop('Patient Id', axis = 1)

#convert non numerical values
df['Level'] = df['Level'].replace(['Low'], '1')
df['Level'] = df['Level'].replace(['Medium'],'2')
df['Level'] = df['Level'].replace(['High'],'3')
df['Level'] = df['Level'].astype(np.int64)
# debugging
# print(df.info())
# #calculate mean and standard deviation
df.agg({'mean','std'})
# #normalise data and print out head
df = (df-df.mean())/df.std()
print(df.agg({'mean','std'}))

df_corr = df.corr()
# print(df_corr.round(2).head(len(df_corr)))
# plt.imshow(df_corr, cmap='RdPu', interpolation='nearest')
sns.heatmap(df_corr,  cmap='RdPu', annot=True, annot_kws={"size":4.5})

#split to test and training set
array = df.values
X = array[:, 0:23]
y = array[:, 23]

# print(X.shape)
# print(y.shape)
# X = df_diabetes['BMI']
# y = df_diabetes['Y']

# print(array)


# test = SelectKBest(score_func=chi2, k=10)
# fit.scores = fit(X,y)
# print(fit.scores_)

# features = fit.transform(X)
# print(features[0:5,:])

# x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.33, random_state=0)

# Model1 = LogisticRegression()
# Model1.fit(x_train, y_train) 

#create new dataframe
# results = pd.DataFrame({'X':df['Age'],'y':y,'lvl_pred':lvl_pred})
# ax1 = results.plot.scatter(x='X', y='y')
# ax2 = results.plot.scatter(x='X', y='lvl_pred', ax=ax1, c='k')

# rmse = (np.sqrt(mean_squared_error(y, y_pred)))
# print(rmse)

#show plots
plt.show()