from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Classify US Representatives into democrats and republicans based on their votes on different issues
# https://archive.ics.uci.edu/ml/index.php

# set plotting style
plt.style.use('ggplot')

# read data
housevotes = pd.read_csv(
    'house-votes-84.data', 
    na_values='?', 
    header=None,
    names=['party', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',\
    'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',\
    'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',\
    'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa'])

# explore data 
housevotes.head()
housevotes.info()
housevotes.dtypes

# feature engineering
housevotes = housevotes.select_dtypes(include=['object']).copy() # make new df with columns of dtype 'object'
housevotes = housevotes[~housevotes.isnull().any(axis=1)] # keep only rows without any null values
housevotes.iloc[:, 3].value_counts() # count no. times each value appears in a column
housevotes = housevotes.replace(['y', 'n'], [1, 0]) # encode 'y' and 'n' as 1 and 0

# train-test split
X = housevotes.drop('party', axis=1)
y = housevotes['party']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state=1, stratify=y)

# kNN prediction
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X, y)
ypred = knn.predict(Xtest)

# model quality
print("Test set predictions:\n {}".format(ypred))
knn.score(Xtest, ytest) # accuracy
precision_score(ytest, ypred, pos_label='democrat') # precision democrats
precision_score(ytest, ypred, pos_label='republican') # precision republicans

# model quality vs. k (no. neighbors)
nneighbors = np.arange(1, 21)
train_acc = []
test_acc = []

for i, k in enumerate(nneighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtrain, ytrain)
    train_acc.append(knn.score(Xtrain, ytrain))
    test_acc.append(knn.score(Xtest, ytest))

# plot accuracy vs. n. neighbors to explore under-/overfitting
plt.figure()
plt.plot(nneighbors, train_acc, label='training')
plt.plot(nneighbors, test_acc, label='test')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks(nneighbors)
plt.legend()
plt.show()


# 2. Classify flower species based on flower morphology (iris dataset)
iris = datasets.load_iris()

irisdf = pd.DataFrame(iris.data, columns=iris.feature_names)
irisdf = irisdf[~irisdf.isnull().any(axis=1)]
Xtrain =irisdf.drop()
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)



# 3. Classify handwritten digits -  MNIST
digits = datasets.load_digits()
digits.DESCR


