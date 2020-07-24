import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from sklearn import metrics, model_selection
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE)
import warnings

# Code to ignore the future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Reading File into data frame
data = pd.read_csv("Breast_cancer_data.csv")

# Splitting the features & target
features = data.iloc[:, :-1]
target = data.diagnosis

# Plot to check the balance between the classes
sns.countplot(x='diagnosis', data=data).set_title("Class Distribution")
plt.show()
# We can see that there is an imbalance of approx 37% - 63%

# We are visualizing the box plot to check the existing outlier
sns.boxplot(data=pd.DataFrame(features)).set_title("Outlier Detection")
plt.show()
# As we can see in the plot, many of the data points are lying as outliers.
# If we remove them much of relevant data is gone. Hence, we are avoiding to remove outlier.

# Dealing with Missing Values
# We will first count number of missing values in each column & target
print("Missing Value in Target", target.isnull().sum())
print("Missing Value in Features\n", features.isnull().sum())
# As the output is coming out to be 0. Hence, No Missing values in features & target.

# Handling Categorical Data
# As my data set have only numeric values, We don't have to handle categorical data

# Finding Correlation for Feature Selection
corr = features.corr()
sns.heatmap(corr).set_title("Correlation in Features")
plt.show()
# mean_radius, mean_perimeter and mean_area are highly correlated
# We can’t reduce our feature space to just three features.
# Hence, we are not extracting these features for classification.

# Scaling Data
scaler = preprocessing.StandardScaler()
features = scaler.fit_transform(features)

# Stratified Cross Fold Validation
skf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Dictionary to store the scores for each model, initialized with empty list to store values for each iteration for each algorithm
model_score_dict = {'knn': [], 'LogReg': [], 'nb': [], 'svm': [], 'rf': [], 'dt': [], 'ab': [], 'gb': []}

# Splitting the data into train & test using Stratified K-Fold Cross Validation
for train_index, test_index in skf.split(features, target):

    # As our data is imbalance we are using SMOTE on training set to over-sample it
    sm = SMOTE(random_state=0)
    X_train, y_train = sm.fit_sample(features[train_index], target[train_index])

    #  ---------------- Implementing Baseline 8 Algorithms ----------------
    # KNN
    knn = neighbors.KNeighborsClassifier()
    knn.fit(X_train, y_train)
    res = knn.predict(features[test_index])
    model_score_dict['knn'].append(metrics.f1_score(res, target[test_index]))

    # Logistic Regression
    LogRegClf = LogisticRegression(random_state=0)
    LogRegClf.fit(X_train, y_train)
    res = LogRegClf.predict(features[test_index])
    model_score_dict['LogReg'].append(metrics.f1_score(res, target[test_index]))
    # Naive Bayes
    NBClf = GaussianNB()
    NBClf.fit(X_train, y_train)
    res = NBClf.predict(features[test_index])
    model_score_dict['nb'].append(metrics.f1_score(res, target[test_index]))
    # SVM
    SVMClf = SVC(gamma="auto", random_state=0)
    SVMClf.fit(X_train, y_train)
    res = SVMClf.predict(features[test_index])
    model_score_dict['svm'].append(metrics.f1_score(res, target[test_index]))
    # Random Forest
    RFClf = RandomForestClassifier(n_estimators=10, random_state=0)
    RFClf.fit(X_train, y_train)
    res = RFClf.predict(features[test_index])
    model_score_dict['rf'].append(metrics.f1_score(res, target[test_index]))
    # Decision Tree Classifier
    DTclf = DecisionTreeClassifier(random_state=0)
    DTclf.fit(X_train, y_train)
    res = DTclf.predict(features[test_index])
    model_score_dict['dt'].append(metrics.f1_score(res, target[test_index]))
    # Ada Boost Classifier
    ABclf = AdaBoostClassifier(random_state=0)
    ABclf.fit(X_train, y_train)
    res = ABclf.predict(features[test_index])
    model_score_dict['ab'].append(metrics.f1_score(res, target[test_index]))
    # Gradient Boosting Classifier
    GBclf = GradientBoostingClassifier(random_state=0)
    GBclf.fit(X_train, y_train)
    res = GBclf.predict(features[test_index])
    model_score_dict['gb'].append(metrics.f1_score(res, target[test_index]))

# Taking mean across all the values through each iteration
score_mean_dict = {k: sum(v) / len(v) for k, v in model_score_dict.items()}

# Print All Algorithms with their Score
for k, v in score_mean_dict.items():
    print("Algorithm:", k, "F1 Score (Averaged over iterations):", v)

# Converting dictionary to DF for plotting in seaborn
# Plot depicts all 8 algorithms with their mean f1-score
score_df = pd.DataFrame.from_dict(score_mean_dict, orient='index', columns=['F1-Score'])
sns.lineplot(data=score_df, markers=True).set_title("Algorithms vs F1-Score")
plt.show()

# Sorting Algos in descending order of values to get the top 3
algo_top = sorted(score_mean_dict, key=score_mean_dict.get, reverse=True)[:3]
print("Top 3 Algorithms with highest accuracy:", algo_top)

# ---------------------------------- Hyper Parameter Optimization --------------------------------------

#  Creating Smote object which will be used as common among all top 3 algo but will be reinitialized every time
# Used to handle imbalance
sm = SMOTE(random_state=0)

# Logistic Regression
print("---Logistic Regression---")
logistic = LogisticRegression()
# Creating a dictionary of hyper parameters, where keys: the parameters & values: the range of values to experiment with
hyperparameters_LR = {'model__C': np.logspace(0, 4, 10), 'model__penalty': ['l1', 'l2']}
# Creating a pipeline of SMOTE and Algorithm
pipeline_LR = Pipeline([('smt', sm), ('model', LogisticRegression())])
# Create grid search using 10-fold cross validation
grid_LR = GridSearchCV(pipeline_LR, hyperparameters_LR, cv=10, verbose=0, scoring='f1')
# Fit grid search to get beat model
best_model_LR = grid_LR.fit(features, target)
# View best hyperparameters
print('Best Parameters:',grid_LR.best_params_) #best_model_LR.best_estimator_.get_params())
print('Best Score:', best_model_LR.best_score_)
#  -----------------------------------------------------------

# ------------------------------------------------------------
# Ada Boosting
print("---Ada Boosting---")
ada_boost = AdaBoostClassifier()

# Creating a dictionary of hyper parameters, where keys: the parameters & values: the range of values to experiment with
hyperparameters_AB = {'model__n_estimators': list(range(50, 101)), 'model__learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9, 1]}
# Creating a pipeline of SMOTE and Algorithm
pipeline_AB = Pipeline([('smt', sm), ('model', ada_boost)])
# Create grid search using 10-fold cross validation
grid_AB = GridSearchCV(pipeline_AB, hyperparameters_AB, cv=10, verbose=0, scoring='f1', n_jobs=-1)
# Fit grid search to get best model
best_model_AB = grid_AB.fit(features, target)
print('Best Parameters:', grid_AB.best_params_)
print('Best Score:', best_model_AB.best_score_)

# -------------------------------------------------------------
# Gradient Boosting
print("---Gradient Boosting---")
grad_boost = GradientBoostingClassifier()
# Creating a dictionary of hyper parameters, where keys: the parameters & values: the range of values to experiment with
hyperparameters_GB = {'model__n_estimators': list(range(100, 200)), 'model__learning_rate': [0.01, 0.03, 0.05, 0.07, 0.09, .1], 'model__max_depth': [1, 3, 5, 7, 9], 'model__random_state': 0}
# Creating a pipeline of SMOTE and Algorithm
pipeline_GB = Pipeline([('smt', sm), ('model', grad_boost)])
# Create grid search using 10-fold cross validation
grid_GB = GridSearchCV(pipeline_GB, hyperparameters_AB, cv=10, verbose=0, scoring='f1', n_jobs=-1)
# Fit grid search to get best model
best_model_GB = grid_GB.fit(features, target)
print('Best Parameters:', grid_GB.best_params_)
print('Best Score:', best_model_GB.best_score_)
#  -------------------------------------------------------------

# ---------------------- Research ----------------------

print("\n\n---------Research: Handling Imbalance Data---------\n")

# Making Object of Stratified K Fold
skf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Initializing dictionaries to store scores
bl_smote_scores = {'LR': 0, 'AB': 0, 'GB': 0}
bl_smote_con_mat = {'LR': [[0, 0], [0, 0]], 'AB': [[0, 0], [0, 0]], 'GB': [[0, 0], [0, 0]]}

km_scores = {'LR': 0, 'AB': 0, 'GB': 0}
km_con_mat = {'LR': [[0, 0], [0, 0]], 'AB': [[0, 0], [0, 0]], 'GB': [[0, 0], [0, 0]]}

svm_sm_scores = {'LR': 0, 'AB': 0, 'GB': 0}

svm_sm_con_mat = {'LR': [[0, 0], [0, 0]], 'AB': [[0, 0], [0, 0]], 'GB': [[0, 0], [0, 0]]}


for train_index, test_index in skf.split(features, target):

    # Borderline Smote
    bl_smote = BorderlineSMOTE(random_state=0, kind='borderline-1')
    X_train, y_train = bl_smote.fit_sample(features[train_index], target[train_index])

    # Logistic Regression
    logistic = LogisticRegression(random_state=0)
    logistic.fit(X_train, y_train)
    res = logistic.predict(features[test_index])
    bl_smote_scores['LR'] += metrics.f1_score(res, target[test_index])
    bl_smote_con_mat['LR'] += confusion_matrix(y_true=target[test_index], y_pred=res)

    # Ada Boost Classifier
    adaBoost = AdaBoostClassifier(random_state=0)
    adaBoost.fit(X_train, y_train)
    res = adaBoost.predict(features[test_index])
    bl_smote_scores['AB'] += metrics.f1_score(res, target[test_index])
    bl_smote_con_mat['AB'] += confusion_matrix(y_true=target[test_index], y_pred=res)

    # Gradient Boost Classifier
    gradBoost = GradientBoostingClassifier(random_state=0)
    gradBoost.fit(X_train, y_train)
    res = gradBoost.predict(features[test_index])
    bl_smote_scores['GB'] += metrics.f1_score(res, target[test_index])
    bl_smote_con_mat['GB'] += confusion_matrix(y_true=target[test_index], y_pred=res)

    # K-Means Smote
    km_smote = KMeansSMOTE(random_state=0)
    X_train, y_train = km_smote.fit_sample(features[train_index], target[train_index])
    # unique, counts = np.unique(y_train, return_counts=True)
    # print("Kmeans uni, count:",np.asarray((unique, counts)).T)

    # Logistic Regression
    logistic = LogisticRegression(random_state=0)
    logistic.fit(X_train, y_train)
    res = logistic.predict(features[test_index])
    km_scores['LR'] += metrics.f1_score(res, target[test_index])
    km_con_mat['LR'] += confusion_matrix(y_true=target[test_index], y_pred=res)
    #
    # Ada Boost Classifier
    adaBoost = AdaBoostClassifier(random_state=0)
    adaBoost.fit(X_train, y_train)
    res = adaBoost.predict(features[test_index])
    km_scores['AB'] += metrics.f1_score(res, target[test_index])
    km_con_mat['AB'] += confusion_matrix(y_true=target[test_index], y_pred=res)

    # Gradient Boost Classifier
    gradBoost = GradientBoostingClassifier(random_state=0)
    gradBoost.fit(X_train, y_train)
    res = gradBoost.predict(features[test_index])
    km_scores['GB'] += metrics.f1_score(res, target[test_index])
    km_con_mat['GB'] += confusion_matrix(y_true=target[test_index], y_pred=res)

    # SVM Smote
    svm_smote = SVMSMOTE(random_state=0)
    X_train, y_train = svm_smote.fit_sample(features[train_index], target[train_index])

    # Logistic Regression
    logistic = LogisticRegression(random_state=0)
    logistic.fit(X_train, y_train)
    res = logistic.predict(features[test_index])
    svm_sm_scores['LR'] += metrics.f1_score(res, target[test_index])
    svm_sm_con_mat['LR'] += confusion_matrix(y_true=target[test_index], y_pred=res)
    #
    # Ada Boost Classifier
    adaBoost = AdaBoostClassifier(random_state=0)
    adaBoost.fit(X_train, y_train)
    res = adaBoost.predict(features[test_index])
    svm_sm_scores['AB'] += metrics.f1_score(res, target[test_index])
    svm_sm_con_mat['AB'] += confusion_matrix(y_true=target[test_index], y_pred=res)

    # Gradient Boost Classifier
    gradBoost = GradientBoostingClassifier(random_state=0)
    gradBoost.fit(X_train, y_train)
    res = gradBoost.predict(features[test_index])
    svm_sm_scores['GB'] += metrics.f1_score(res, target[test_index])
    svm_sm_con_mat['GB'] += confusion_matrix(y_true=target[test_index], y_pred=res)


bl_smote_scores = {k: v / 10 for k, v in bl_smote_scores.items()}
bl_smote_con_mat = {k: v for k, v in bl_smote_con_mat.items()}
print("Borderline Smote Score:", bl_smote_scores)
print("Borderline Smote Con Mat Score\n:", bl_smote_con_mat)
bl_smote_df = pd.DataFrame.from_dict(bl_smote_scores, orient='index', columns=['F1-Score'])
sns.lineplot(data=bl_smote_df, markers=True).set_title("Using Border Line Smote")
plt.show()


km_scores = {k: v / 10 for k, v in km_scores.items()}
km_con_mat = {k: v / 10 for k, v in km_con_mat.items()}
print("K Means Smote Score:", km_scores)
print("K Means Smote Con Mat Score:\n", km_con_mat)
km_df = pd.DataFrame.from_dict(km_scores, orient='index', columns=['F1-Score'])
sns.lineplot(data=km_df, markers=True).set_title("Using K Means Smote")
plt.show()

svm_sm_scores = {k: v / 10 for k, v in svm_sm_scores.items()}
svm_sm_con_mat = {k: v / 10 for k, v in svm_sm_con_mat.items()}
print("SVM Smote Score:", svm_sm_scores)
print("SVM Smote Con Mat Score:\n", svm_sm_con_mat)
svm_sm_df = pd.DataFrame.from_dict(svm_sm_scores, orient='index', columns=['F1-Score'])
sns.lineplot(data=svm_sm_df, markers=True).set_title("Using SVM Smote")
plt.show()


# ------------ Hyper Parameter Optimization with SMOTE Variants -----------
# ----------- Borderline Smote -----------
print("\n\nHyper Parameter Optimization with SMOTE Variants")
# Logistic Regression + Borderline Smote
print("\n---Logistic Regression + Borderline---")
logistic = LogisticRegression()
bl_smote = BorderlineSMOTE(random_state=0, kind='borderline-1')
# Creating a dictionary of hyper parameters, where keys: the parameters & values: the range of values to experiment with
hyperparameters_LR = {'model__C': np.logspace(0, 4, 10), 'model__penalty': ['l1', 'l2']}
# Creating a pipeline of SMOTE and Algorithm
pipeline_LR = Pipeline([('smt', bl_smote), ('model', LogisticRegression())])
# Create grid search using 10-fold cross validation
grid_LR = GridSearchCV(pipeline_LR, hyperparameters_LR, cv=10, verbose=0, scoring='f1')
# Fit grid search to get beat model
best_model_LR = grid_LR.fit(features, target)
# View best hyperparameters
print('Best Parameters:', grid_LR.best_params_)
print('Best Score:', best_model_LR.best_score_)
# ------------------------------------------------------------
# Ada Boosting + BL Smote
print("\n---Ada Boosting + BorderlineSMOTE---")
ada_boost = AdaBoostClassifier()
bl_smote = BorderlineSMOTE(random_state=0, kind='borderline-1')
# Creating a dictionary of hyper parameters, where keys: the parameters & values: the range of values to experiment with
hyperparameters_AB = {'model__n_estimators': list(range(50, 101)), 'model__learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9, 1]}
# Creating a pipeline of SMOTE and Algorithm
pipeline_AB = Pipeline([('smt', bl_smote), ('model', ada_boost)])
# Create grid search using 10-fold cross validation
grid_AB = GridSearchCV(pipeline_AB, hyperparameters_AB, cv=10, verbose=0, scoring='f1', n_jobs=-1)
# Fit grid search to get best model
best_model_AB = grid_AB.fit(features, target)
print('Best Parameters:', grid_AB.best_params_)
print('Best Score:', best_model_AB.best_score_)

# -------------------------------------------------------------
# Gradient Boosting + K Means Smote
print("\n---Gradient Boosting + BorderlineSMOTE---")
grad_boost = GradientBoostingClassifier()
bl_smote = BorderlineSMOTE(random_state=0, kind='borderline-1')
# Creating a dictionary of hyper parameters, where keys: the parameters & values: the range of values to experiment with
hyperparameters_GB = {'model__n_estimators': list(range(100, 200)), 'model__learning_rate': [0.01, 0.03, 0.05, 0.07, 0.09, .1], 'model__max_depth': [1, 3, 5, 7, 9], 'model__random_state': 0}
# Creating a pipeline of SMOTE and Algorithm
pipeline_GB = Pipeline([('smt', bl_smote), ('model', grad_boost)])
# Create grid search using 10-fold cross validation
grid_GB = GridSearchCV(pipeline_GB, hyperparameters_AB, cv=10, verbose=0, scoring='f1', n_jobs=-1)
# Fit grid search to get best model
best_model_GB = grid_GB.fit(features, target)
print('Best Parameters:',grid_GB.best_params_)
print('Best Score:', best_model_GB.best_score_)

# ----------- K Means Smote -----------
# Logistic Regression + K Means Smote
print("\n---Logistic Regression + K Means Smote ---")
logistic = LogisticRegression()
km_smote = KMeansSMOTE(random_state=0)
# Creating a dictionary of hyper parameters, where keys: the parameters & values: the range of values to experiment with
hyperparameters_LR = {'model__C': np.logspace(0, 4, 10), 'model__penalty': ['l1', 'l2']}
# Creating a pipeline of SMOTE and Algorithm
pipeline_LR = Pipeline([('smt', km_smote), ('model', LogisticRegression())])
# Create grid search using 10-fold cross validation
grid_LR = GridSearchCV(pipeline_LR, hyperparameters_LR, cv=10, verbose=0, scoring='f1')
# Fit grid search to get beat model
best_model_LR = grid_LR.fit(features, target)
# View best hyperparameters
print('Best Parameters:',grid_LR.best_params_)
print('Best Score:', best_model_LR.best_score_)

# ------------------------------------------------------------
# Ada Boosting + K means Smote
print("\n---Ada Boosting + K Means Smote ---")
ada_boost = AdaBoostClassifier()
km_smote = KMeansSMOTE(random_state=0)
# Creating a dictionary of hyper parameters, where keys: the parameters & values: the range of values to experiment with
hyperparameters_AB = {'model__n_estimators': list(range(50, 101)), 'model__learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9, 1]}
# Creating a pipeline of SMOTE and Algorithm
pipeline_AB = Pipeline([('smt', km_smote), ('model', ada_boost)])
# Create grid search using 10-fold cross validation
grid_AB = GridSearchCV(pipeline_AB, hyperparameters_AB, cv=10, verbose=0, scoring='f1', n_jobs=-1)
# Fit grid search to get best model
best_model_AB = grid_AB.fit(features, target)
print('Best Parameters:',grid_AB.best_params_)
print('Best Score:', best_model_AB.best_score_)

# -------------------------------------------------------------
# Gradient Boosting + K Means Smote
print("\n---Gradient Boosting + K Means Smote ---")
grad_boost = GradientBoostingClassifier()
km_smote = KMeansSMOTE(random_state=0)
# Creating a dictionary of hyper parameters, where keys: the parameters & values: the range of values to experiment with
hyperparameters_GB = {'model__n_estimators': list(range(100, 200)), 'model__learning_rate': [0.01, 0.03, 0.05, 0.07, 0.09, .1], 'model__max_depth': [1, 3, 5, 7, 9], 'model__random_state': 0}
# Creating a pipeline of SMOTE and Algorithm
pipeline_GB = Pipeline([('smt', km_smote), ('model', grad_boost)])
# Create grid search using 10-fold cross validation
grid_GB = GridSearchCV(pipeline_GB, hyperparameters_AB, cv=10, verbose=0, scoring='f1', n_jobs=-1)
# Fit grid search to get best model
best_model_GB = grid_GB.fit(features, target)
print('Best Parameters:', grid_GB.best_params_)
print('Best Score:', best_model_GB.best_score_)


# ---------------- SVM Smote ---------------
# Logistic Regression + SVM Smote
print("\n---Logistic Regression + SVM Smote---")
logistic = LogisticRegression()
svm_smote = SVMSMOTE(random_state=0)
# Creating a dictionary of hyper parameters, where keys: the parameters & values: the range of values to experiment with
hyperparameters_LR = {'model__C': np.logspace(0, 4, 10), 'model__penalty': ['l1', 'l2']}
# Creating a pipeline of SMOTE and Algorithm
pipeline_LR = Pipeline([('smt', svm_smote), ('model', LogisticRegression())])
# Create grid search using 10-fold cross validation
grid_LR = GridSearchCV(pipeline_LR, hyperparameters_LR, cv=10, verbose=0, scoring='f1')
# Fit grid search to get beat model
best_model_LR = grid_LR.fit(features, target)
# View best hyperparameters
print('Best Parameters:',grid_LR.best_params_)
print('Best Score:', best_model_LR.best_score_)
#  -----------------------------------------------------------

# ------------------------------------------------------------
# Ada Boosting + SVM Smote
print("\n---Ada Boosting + SVM Smote---")
ada_boost = AdaBoostClassifier()
svm_smote = SVMSMOTE(random_state=0)
# Creating a dictionary of hyper parameters, where keys: the parameters & values: the range of values to experiment with
hyperparameters_AB = {'model__n_estimators': list(range(50, 101)), 'model__learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9, 1]}
# Creating a pipeline of SMOTE and Algorithm
pipeline_AB = Pipeline([('smt', svm_smote), ('model', ada_boost)])
# Create grid search using 10-fold cross validation
grid_AB = GridSearchCV(pipeline_AB, hyperparameters_AB, cv=10, verbose=0, scoring='f1', n_jobs=-1)
# Fit grid search to get best model
best_model_AB = grid_AB.fit(features, target)
print('Best Parameters:',grid_AB.best_params_)
print('Best Score:', best_model_AB.best_score_)

# -------------------------------------------------------------
# Gradient Boosting + SVM Smote
print("\n---Gradient Boosting + SVM Smote---")
grad_boost = GradientBoostingClassifier()
svm_smote = SVMSMOTE(random_state=0)
# Creating a dictionary of hyper parameters, where keys: the parameters & values: the range of values to experiment with
hyperparameters_GB = {'model__n_estimators': list(range(100, 200)), 'model__learning_rate': [0.01, 0.03, 0.05, 0.07, 0.09, .1], 'model__max_depth': [1, 3, 5, 7, 9], 'model__random_state': 0}
# Creating a pipeline of SMOTE and Algorithm
pipeline_GB = Pipeline([('smt', svm_smote), ('model', grad_boost)])
# Create grid search using 10-fold cross validation
grid_GB = GridSearchCV(pipeline_GB, hyperparameters_GB, cv=10, verbose=0, scoring='f1', n_jobs=-1)
# Fit grid search to get best model
best_model_GB = grid_GB.fit(features, target)
print('Best Parameters:',grid_GB.best_params_)
print('Best Score:', best_model_GB.best_score_)

