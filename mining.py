__author__ = 'Horace'

import numpy, pandas
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn import naive_bayes
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.metrics import confusion_matrix, roc_curve, f1_score

from sklearn.externals.six import StringIO
import pydot
from sklearn.learning_curve import learning_curve, validation_curve
import matplotlib.pyplot as plt


# read data
## 1309 x 27
titanicDf = pandas.read_csv("titanic_data.csv")
titanicDf = titanicDf.drop('index',1)




def dataCleansing(turn_on, titanicDf):
    if turn_on == 1:
        # Impute missing value of Age
        linModel = linear_model.LinearRegression()

        knownDf = titanicDf[titanicDf.age.notnull()]
        knownCol = [kc for kc in knownDf.columns if kc not in ['age', 'Survival']]  # exclude survival
        knownDfExcludeAge = knownDf[knownCol]

        linModel.fit(knownDfExcludeAge.values, knownDf.age.values)

        imputedAge = linModel.predict(titanicDf[knownCol].values)
        titanicDf['age'] = titanicDf['age'].fillna(pandas.Series(imputedAge))
    else:
        titanicDf = titanicDf[titanicDf.age.notnull()]
    return titanicDf

# Setting Data Cleansing
titanicDf = dataCleansing(turn_on=1, titanicDf=titanicDf)
#titanicDf = titanicDf.drop('age', 1)


"""
titanicDf['class_1', 'class_2', 'class_3', 'male', 'female', 'age', 'sipsp',
       'parch', 'A_ticket', 'C_ticket', 'CA_ticket', 'FC_ticket',
       'numeric_ticket', 'Others_ticket', 'PC_ticket', 'PP_ticket',
       'SC_ticket', 'SO_ticket', 'SOTON_ticket', 'STON_ticket', 'W_ticket',
       'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Survival']



titanicDf = titanicDf[['class_1', 'class_3', 'male', 'age',
       'Fare', 'Embarked_C', 'Embarked_S', 'Survival']]
"""

# train and test data partition
## 891 x 27
trainDf = titanicDf[titanicDf.Survival.notnull()]

# 418 x 27
testDf = titanicDf[titanicDf.Survival.isnull()]
testDf = testDf.drop('Survival',1)

inputDf = trainDf.ix[:,trainDf.columns != "Survival"]
targetDf = trainDf["Survival"]

# Dimension of training set
print("Dimension of training set: ")
print(trainDf.shape)


print("Dimension of testing set: ")
print(testDf.shape)

X = inputDf.as_matrix()
y = targetDf.as_matrix()

# Assessment function
def accuracy(y_actual, y_pred):
    acc = 0
    for i in range(1,len(y)):
        if y_actual[i] == y_pred[i]:
            acc = acc + 1
    acc_rate = acc / float(len(y_actual))
    return round(acc_rate,2)

# Plot Learning Curve
def plot_learning_curve(classifier, X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Learning Curve")
    plt.ylim((0,1))
    plt.xlabel("Training Example")
    plt.ylabel("Score")
    train_sizes, train_scores, validation_scores = learning_curve(classifier, X, y)
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    validation_scores_mean = numpy.mean(validation_scores, axis=1)
    validation_scores_std = numpy.std(validation_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="g")
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.1, color="r")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="g", label="Training Score")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="r", label="Cross-validation Score")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()

"""
# Plot data instances in 2-D graph
pca = PCA(n_components=2)
pca.fit(X)
X_transform = pca.transform(X)
pc1 = pandas.Series(X_transform[:,0])
pc2 = pandas.Series(X_transform[:,1])
titanicDf['pc1'] = pc1
titanicDf['pc2'] = pc2
titanicDf.to_csv("titanic_pca.csv")

fig = plt.figure()
ax = plt.subplot(111)
color = trainDf['class_3'].as_matrix()

import matplotlib.cm as cm
plt.scatter(pc1.values, pc2.values, c=y, cmap="winter_r", label='Survived')
plt.legend(prop={'size':12}, loc='center right', bbox_to_anchor=(1, 0.5))
plt.show()



# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
# factorColor = ['g','b','black','red']
# factors = pca.components_
# eigenvalues = pca.explained_variance_ratio_
# for i in range(len(X[1])):
#     x = numpy.arange(0, 400, 20) * factors[0][i]
#     y = numpy.arange(0, 400, 20) * factors[1][i]
#     plt.plot(x, y)#, color=factorColor[i], lw=2) #, label=abilityCol[i])

plt.show()
"""

"""
# Decision Tree
ctreeModel = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=1, min_samples_leaf=4)
ctreeModel.fit(X,y)
y_ctree_pred = ctreeModel.predict(X)

dot_data = StringIO()
tree.export_graphviz(ctreeModel, feature_names=inputDf.columns,  out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("ctree.pdf")

print "Classification Tree:"
print "Accuracy: ", accuracy(y,y_ctree_pred)
print confusion_matrix(y, y_ctree_pred, labels=[1,0])
#print logModel.decision_function(X)
#plot_learning_curve(ctreeModel, X, y)
print "F1 score: ", f1_score(y, y_ctree_pred)

print "\n"

# Logistic Regression Model
logModel = linear_model.LogisticRegression(C=3, solver='liblinear')
logModel.fit(X,y)
y_log_pred = logModel.predict(X)

print logModel.coef_
print titanicDf.columns
print logModel.intercept_

print "Logistics Regression:"
print "Accuracy: ", accuracy(y,y_log_pred)
print confusion_matrix(y, y_log_pred, labels=[1,0])
#print logModel.decision_function(X)
#plot_learning_curve(logModel, X, y)
print "F1 score: ", f1_score(y, y_log_pred)

print "\n"

# Support Vector Machine Classifier
svmModel = svm.SVC(C=1, kernel="rbf")
svmModel.fit(X,y)
y_svm_pred = svmModel.predict(X)
#print svmModel.score(X,y)
print "SVM:"
print "Accuracy: ", accuracy(y,y_svm_pred)
print confusion_matrix(y, y_svm_pred, labels=[1,0])
#print svmModel.decision_function(X)
#plot_learning_curve(svmModel, X, y)
print "F1 score: ", f1_score(y, y_svm_pred)

print "\n"


# Bayes Classifier
bayesModel = naive_bayes.GaussianNB()
bayesModel.fit(X,y)
y_bayes_pred = bayesModel.predict(X)

print "Bayes Classifier:"
#print bayesModel.decision_function(X)
print "Accuracy: ", accuracy(y,y_bayes_pred)
#print confusion_matrix(y, y_bayes_pred, labels=[1,0])
#plot_learning_curve(bayesModel, X, y)
print "F1 score: ", f1_score(y, y_bayes_pred)


"""

# Gradient Boosting Classifier
clf = GradientBoostingClassifier(loss="deviance", n_estimators=500, max_depth=3, learning_rate=0.05)
#clf = RandomForestClassifier(criterion="gini", max_depth=5, class_weight="balanced")
clf.fit(X,y)
y_gdc_pred = clf.predict(X)


"""
# ROC Curve
modelList = ["CTree", "LogReg", "SVC", "Bayes"]
dec_func_list = list()
dec_func_list.append(ctreeModel.predict_proba(X)[:,1])
dec_func_list.append(logModel.decision_function(X))
dec_func_list.append(svmModel.decision_function(X))
dec_func_list.append(bayesModel.predict_proba(X)[:,1])

#print dec_func_list[1]
plt.figure()
#print ctreeModel.predict_proba(X)[:,0]
for i in range(4):
    fpr, tpr, thresholds = roc_curve(y, dec_func_list[i])
    plt.plot(fpr, tpr, label=modelList[i] )  # plot the false positive and true positive rate
    plt.plot([0, 1], [0, 1], 'k--')  # plot the threshold
    plt.xlim([0.0, 1.0])  # limit x axis
    plt.ylim([0.0, 1.05])  # limit y axis
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

#plt.show()

"""


def toCsv(list):
    df = pandas.concat([pandas.Series(range(892,1310), name="PassengerId"), pandas.Series(list, dtype=int, name="Survived")], axis=1)
    df.to_csv("submission.csv", index=False)

# testing data
X_test = testDf.values
y_test_pred = clf.predict(X_test)
toCsv(y_test_pred)

print(len(y))
print(confusion_matrix(y, y_gdc_pred))
print(clf.score(X,y))

pos = 0
neg = 0
for i in y_test_pred:
    if i == 1:
        pos = pos + 1
    else:
        neg = neg + 1

print(pos)
print(neg)

print(pos/(pos+neg))
#print(confusion_matrix(y_test_pred))


print(titanicDf.columns)