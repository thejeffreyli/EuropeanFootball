import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

from sklearn import metrics


"KNN"
def knn(xTrain, yTrain):
    # from sklearn document on LogisticRegression
    neigh = KNeighborsClassifier(n_neighbors=5, metric='euclidean').fit(xTrain, yTrain) 
    return neigh


"LOGISTIC REGRESSION"
def logReg(xTrain, yTrain):
    # from sklearn document on LogisticRegression
    clf = LogisticRegression(random_state=0).fit(xTrain, yTrain) 
    return clf

"GAUSSIAN NAIVE BAYES"
def gaussNB(xTrain, yTrain):
    # from sklearn document on GaussianNB
    clf = GaussianNB().fit(xTrain, yTrain)
    return clf

"RANDOM FOREST"
def randForest(xTrain, yTrain):
    clf = RandomForestClassifier().fit(xTrain, yTrain)
    return clf

"ADABOOST"
def ada(xTrain, yTrain):
    clf = AdaBoostClassifier(n_estimators = 100, random_state = 0).fit(xTrain, yTrain)
    return clf

# read an input file and convert it to numpy
def file_to_numpy(filename):
    df = pd.read_csv(filename)
    return df.to_numpy()

"*******************Preprocessing*******************"

os.chdir("C:/Users/hp/Desktop/MLFall20/draft")

# PLEASE READ
# all.csv was created by joining together home.csv, away.csv, and labels.csv
# merge function was not able to combine them so this was done manually
dat = pd.read_csv("all.csv")

y = dat["label"]
x = dat.drop(['label'], axis=1) 

# pca = PCA()
# x = pca.fit(x).transform(x)

lda =  LinearDiscriminantAnalysis()
x = lda.fit(x,y).transform(x)


xTrain, xTest, yTrain, yTest = train_test_split(
     x, y, test_size=0.33, random_state=42)


"*******************Testing Models*******************"
"*******************Generating Confusion Matrices and Plots*******************"

"Logistic Regression (Binary)"

lr = logReg(xTrain, yTrain)

print("Logistic Regression train set score: ", accuracy_score(yTrain, lr.predict(xTrain)))
tn, fp, fn, tp = confusion_matrix(yTrain, lr.predict(xTrain)).ravel()
print("Confusion Matrix Results: ", tn, fp, fn, tp)

print("Logistic Regression test set score: ", accuracy_score(yTest, lr.predict(xTest)))
tn, fp, fn, tp = confusion_matrix(yTest, lr.predict(xTest)).ravel()
print("Confusion Matrix Results: ", tn, fp, fn, tp)


disp = plot_confusion_matrix(lr, xTest, yTest, cmap=plt.cm.Blues)
disp.ax_.set_title("Logistic Regression")

  # doctest: +SKIP


"Gaussian Naive Bayes"

gNB = gaussNB(xTrain, yTrain)

print("Gaussian Naive Bayes train set score: ", accuracy_score(yTrain, gNB.predict(xTrain)))
tn, fp, fn, tp = confusion_matrix(yTrain, gNB.predict(xTrain)).ravel()
print("Confusion Matrix Results: ", tn, fp, fn, tp)

print("Gaussian Naive Bayes test set score: ", accuracy_score(yTest, gNB.predict(xTest)))
tn, fp, fn, tp = confusion_matrix(yTest, gNB.predict(xTest)).ravel()
print("Confusion Matrix Results: ", tn, fp, fn, tp)

disp = plot_confusion_matrix(gNB, xTest, yTest, cmap=plt.cm.Blues)
disp.ax_.set_title("Gaussian Bayes")



"KNN"

kNN = knn(xTrain, yTrain)

print("KNN train set score: ", accuracy_score(yTrain, kNN.predict(xTrain)))
tn, fp, fn, tp = confusion_matrix(yTrain, kNN.predict(xTrain)).ravel()
print("Confusion Matrix Results: ", tn, fp, fn, tp)

print("KNN test set score: ", accuracy_score(yTest, kNN.predict(xTest)))
tn, fp, fn, tp = confusion_matrix(yTest, kNN.predict(xTest)).ravel()
print("Confusion Matrix Results: ", tn, fp, fn, tp)

disp = plot_confusion_matrix(kNN, xTest, yTest, cmap=plt.cm.Blues)
disp.ax_.set_title("KNN")


"Random Forest"

randomForest = randForest(xTrain, yTrain)

print("Random Forest train set score: ", accuracy_score(yTrain, randomForest.predict(xTrain)))
tn, fp, fn, tp = confusion_matrix(yTrain, randomForest.predict(xTrain)).ravel()
print("Confusion Matrix Results: ", tn, fp, fn, tp)

print("Random Forest test set score: ", accuracy_score(yTest, randomForest.predict(xTest)))
tn, fp, fn, tp = confusion_matrix(yTest, randomForest.predict(xTest)).ravel()
print("Confusion Matrix Results: ", tn, fp, fn, tp)

disp = plot_confusion_matrix(randomForest, xTest, yTest, cmap=plt.cm.Blues)
disp.ax_.set_title("Random Forest")


"Adaboost"

adaboost = ada(xTrain, yTrain)

print("Adaboost train set score: ", accuracy_score(yTrain, adaboost.predict(xTrain)))
tn, fp, fn, tp = confusion_matrix(yTrain, adaboost.predict(xTrain)).ravel()
print("Confusion Matrix Results: ", tn, fp, fn, tp)

print("Adaboost test set score: ", accuracy_score(yTest, adaboost.predict(xTest)))
tn, fp, fn, tp = confusion_matrix(yTest, adaboost.predict(xTest)).ravel()
print("Confusion Matrix Results: ", tn, fp, fn, tp)

disp = plot_confusion_matrix(adaboost, xTest, yTest, cmap=plt.cm.Blues)
disp.ax_.set_title("Adaboost")

a = metrics.plot_roc_curve(lr, xTest, yTest)
b = metrics.plot_roc_curve(gNB, xTest, yTest) 
c = metrics.plot_roc_curve(kNN, xTest, yTest) 
d = metrics.plot_roc_curve(randomForest, xTest, yTest) 
e = metrics.plot_roc_curve(adaboost, xTest, yTest) 

