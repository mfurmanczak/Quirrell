#import pandas for data processing
import pandas as pd
#import numpy for matrices
import numpy as np
#import matplotlib for ploting graphs
import matplotlib.pyplot as plt
#import sklearn features used for calculating RMSE, fitting regression model and Polynomial features
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#import seaborn for heatmaps
import seaborn as sns
#import sklearn features for score functions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
#import classification report
from sklearn.metrics import classification_report
#import KFold for tuning tests
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#import accuracy_score to retrieve accuracy from predictions
from sklearn.metrics import accuracy_score

#read xlsx file/create dataframe
df = pd.read_excel('data/datasets.xlsx')
#remove unnecessary fields from the dataframe
df = df.drop('Patient Id', axis = 1)

#convert non numerical values
df['Level'] = df['Level'].replace(['Low'], '1')
df['Level'] = df['Level'].replace(['Medium'],'2')
df['Level'] = df['Level'].replace(['High'],'3')
df['Level'] = df['Level'].astype(np.int64)
# debugging
# print(df.head())
#calculate mean and standard deviation
df_aggr = df.agg({'mean','std'})
#normalise data and print out head
newdata = (df-df.mean())/df.std()
#print(df.agg({'mean','std'}))

newdata = df.corr()
#print(df_corr.round(2).head(len(df_corr)))
# plt.imshow(newdata, cmap='RdPu', interpolation='nearest')
cmap = sns.diverging_palette(240,240, as_cmap=True)
sns.heatmap(newdata, cmap=cmap, annot=True, annot_kws={"size":4.5})

#split to input and output
array = df.values
X = array[:, 0:23]
y = array[:, 23]
# print(array)
# print(X.shape)
# print(y)

#array to store the indexes and corresponding accuracies
indexes = []
accuracies = []

for i in range(1,24):
    #open file for results
    
    bestfeatures = SelectKBest(score_func=chi2, k=i)
    fit = bestfeatures.fit(X,y)
    # print(fit.scores_)
    # print(fit.pvalues_)

    bestfeatures = fit.transform(X)
    # print(get_feature_names_out(input_features=None))
    # print(bestfeatures[0:10,:])
    # print(bestfeatures.shape)

    #split into test and training set
    x_train, x_test, y_train, y_test = train_test_split(bestfeatures, y, test_size=0.33, random_state=0)

    # print(np.info(object=bestfeatures))

    #build the model
    Model1 = LogisticRegression(solver='liblinear', random_state=0)
    Model1.fit(x_train, y_train)

    score = Model1.score(x_test, y_test)
    #make predictions
    predictions1 = Model1.predict(x_test)
    # print(predictions1)
    mse = mean_absolute_error(y_test, predictions1)
    # print(mse)
    indexes.append(i)
    accuracies.append(accuracy_score(y_test, predictions1))
    # print(classification_report(y_test, predictions1))

#register results into results file
    # f = open("resultschi2.txt", "a")
    # f.write("****************CLASSIFICATION REPORT****************\n")
    # f.write("*****************************************************\n")
    # f.write("k = "+ str(i))    
    # f.write("\n")
    # f.write(classification_report(y_test, predictions1))
    # f.write("\n")
    # f.close()

#accuracy scattergram and plot
# sc = pd.DataFrame({'accuracies':accuracies, 'indexes':indexes})
# sc.plot(x='indexes', y='accuracies')
# sc.plot.scatter(x='indexes', y='accuracies')
# plt.scatter(indexes,accuracies)

#test options and evaluation metric
# num_folds = 5
# seed = 3
# scoring = 'accuracy'

# kfold = KFold(n_splits=num_folds, random_state=None)
# cv_results = cross_val_score(Model1, x_train, y_train, scoring='accuracy', cv=kfold)
# msg = '%f (%f)'%(cv_results.mean(), cv_results.std())
# print(msg)

#plot scattergram to verify relevancy of the results
# results = pd.DataFrame({'y_test':y_test, 'predictions1':predictions1})
# results.plot.scatter(x='y_test', y='predictions1')

#show plots
plt.show()