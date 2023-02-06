import pandas as pd
import seaborn as sns
import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import sklearn.metrics as metrics
from joblib import parallel_backend
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import matplotlib
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('axes', labelsize=9)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=9)    # fontsize of the tick labels
plt.rc('ytick', labelsize=9)    # fontsize of the tick labels
plt.rc('legend', fontsize=9)    # legend fontsize
plt.rc('font', size=5)          # controls default text sizes

matplotlib.rcParams.update({'font.size':7 })
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"



employeesdf = pd.read_csv('employeesdf.csv').drop(['Unnamed: 0'],axis=1)

integer_columns = employeesdf.select_dtypes('int64')
for c in integer_columns:
    employeesdf[c] = employeesdf[c].astype('int32')

float_columns = employeesdf.select_dtypes('float64')
for c in float_columns:
    employeesdf[c] = employeesdf[c].astype('float32')

mldf = employeesdf.copy()
#Label Encoding
le = LabelEncoder()

columns_to_encode = ['MonthlyIncome','Gender','Department','BusinessTravel']

for columns in columns_to_encode:
    mldf[columns] = le.fit_transform(mldf[columns])

#Replacing nan values with mean to prevent data loss
mean_complaintyears = mldf['complaintyears'].mean()
mldf['complaintyears'].fillna(mean_complaintyears, inplace=True)

#Print nan columns
# print(np.isnan(mldf).any())

#Oversample the data to create equal distribution of Left class

X = mldf.drop(['Left'], axis=1)
y = mldf['Left'].astype(int)


#ros = RandomOverSampler(sampling_strategy=1) # Float
ros = RandomOverSampler(sampling_strategy= 1) # String
X_res, y_res = ros.fit_resample(X, y)



#Turning the resampled data into a dataframe
resampled_df = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)
resampled_df.reset_index(drop=True, inplace=True)



X = resampled_df.drop(columns=['Left'])
y = resampled_df['Left']

# Define the ensemble method
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)

# fit the classifier
clf2.fit(X, y)

# Compute the cumulative distribution function gradient
cdf_gradient = np.gradient(np.cumsum(clf2.feature_importances_))

# Select features using the ensemble method
sfm = SelectFromModel(estimator=clf2, threshold=np.percentile(cdf_gradient, 50))
sfm.fit(X, y)
X_transform = sfm.transform(X)

# Print the list of selected features
selected_features = X.columns[sfm.get_support()]
print(selected_features)



importances = sfm.estimator_.feature_importances_[sfm.get_support()]
sorted_indices = np.argsort(-importances)
plt.barh(np.array(selected_features)[sorted_indices], importances[sorted_indices])
plt.title('Selected Features and Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
#
# def logisticreg(data, top_n):
#     top_features = sorted(selected_features, key=lambda x: -len(x))[:top_n]
#     X = data[top_features]
#     y = data['Left']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
#
#     lr = LogisticRegression(max_iter=10000)
#     lr.fit(X_train, y_train)
#
#     y_pred = lr.predict(X_test)
#
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     accuracy = accuracy_score(y_test, y_pred)
#
#     return precision, recall, f1, accuracy
#
# list = []
# for i in range(1,20,1):
#         precision, recall, f1, accuracy = logisticreg(resampled_df, i)
#         print("Performance for Logistic Model with Top {} features is precision : {}, recall : {}, f1 score : {}, accuracy : {}".format(i, precision, recall, f1, accuracy))
#         list.append([i, precision, recall, f1, accuracy])
#
# df = pd.DataFrame(list, columns=['num_of_features','precision', 'recall', 'f1_score', 'accuracy'])
# print(df)
#
# sns.lineplot(x='num_of_features', y='precision', data=df, label='Precision Score')
# sns.lineplot(x='num_of_features', y='recall', data=df, label='Recall Score')
# sns.lineplot(x='num_of_features', y='f1_score', data=df, label='F1 Score')
# sns.lineplot(x='num_of_features', y='accuracy', data=df, label='Acc Score')
# plt.title('Logistic regression Performance')
# plt.show()
#
# def random_forest(data, top_n, n_estimators=100, max_depth=None):
#     top_features = sorted(selected_features, key=lambda x: -len(x))[:top_n]
#     X = data[top_features]
#     y = data['Left']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
#
#     rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
#     rf.fit(X_train, y_train)
#
#     y_pred = rf.predict(X_test)
#
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     accuracy = accuracy_score(y_test, y_pred)
#
#     return precision, recall, f1, accuracy
#
# # Train and test the model with different number of features
# list = []
# for i in range(1, 20, 1):
#     precision, recall, f1, accuracy = random_forest(resampled_df, i)
#     print("Performance for Random Forest with Top {} features is precision : {}, recall : {}, f1 score : {}, accuracy : {}".format(i, precision, recall, f1, accuracy))
#     list.append([i, precision, recall, f1, accuracy])
#
# # Create a dataframe to hold the results and visualize the performance
# df = pd.DataFrame(list, columns=['num_of_features','precision', 'recall', 'f1_score', 'accuracy'])
# print(df)
#
# sns.lineplot(x='num_of_features', y='precision', data=df, label='Precision Score')
# sns.lineplot(x='num_of_features', y='recall', data=df, label='Recall Score')
# sns.lineplot(x='num_of_features', y='f1_score', data=df, label='F1 Score')
# sns.lineplot(x='num_of_features', y='accuracy', data=df, label='Acc Score')
# plt.title('Random Forest Classifer Performace')
# plt.show()