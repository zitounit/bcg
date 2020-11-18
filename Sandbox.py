import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
import pprint as pp
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

"""
print("Starting")
#df = pd.read_csv("data/customer_data.csv")
df = pd.read_csv("/Users/alwo/Programming/Gammathon/data/c_data_sample.csv")
df_nchurned = pd.read_csv("/Users/alwo/Programming/Gammathon/data/c_data_sample.csv", nrows=400)[df["churned"] == False]
#df_nchurned = df_nchurned[df_nchurned["earned_reward_points"] < df_nchurned["earned_reward_points"].confidence(0.95)]
df_churned = df[df["churned"] == True]
#df_churned = df_churned[df_churned["earned_reward_points"] < df_churned["earned_reward_points"].confidence(0.95)]
df_sample = df_nchurned.append(df_churned)
print("Done")
df_sample.append(df_nchurned)

sn.pairplot(df_sample, hue="churned")
plt.savefig("figure.png", quality=30)
"""
print("Start")
nrows = 10000

#Read data 
df = pd.read_csv("/Users/alwo/Programming/Gammathon/data/customer_data.csv", nrows=nrows)

#Cut data to remove outliers
df2 = df[df["earned_reward_points"] <= 2000]
df2 = df2[df2["referred_friends"] <= 6]
X = df.drop(columns=["churned"])
X = X.drop(columns=["month"])
X = X.drop(columns=["cluster"])
y = df["churned"].values
#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 6)
# Fit the classifier to the data
knn.fit(X_train,y_train)

knn.predict(X_test)[0:5]

#check accuracy of our model on the test data
knn.score(X_test, y_test)

from sklearn.model_selection import cross_val_score
import numpy as np
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X, y, cv=5)
#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

from sklearn.model_selection import GridSearchCV
#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, y)
#check top performing n_neighbors value
knn_gscv.best_params_
#check mean score for the top performing value of n_neighbors
knn_gscv.best_score_

print("done")