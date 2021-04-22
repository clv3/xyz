import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
import pickle
# warnings.filterwarnings("ignore")

data = pd.read_csv("teamData.csv")
data = np.array(data)

X = data[:, [0,1,2,3,4,5,6,7,8,9,10,11]]
y = data[:, -1]
X = X.astype('int')

le = LabelEncoder()
y=le.fit_transform(y)
# print(X[0], y[0])

# splits data to training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Creates logistic regression classifier
model = LogisticRegression(max_iter=1200)

#Train the classifier with the training dataset
model.fit(X_train, y_train)

# # joblib.dump(model, "WOWCHA.joblib")
# # model = joblib.load("WOWCHA.joblib")

# Pass the dataset that contains input values for testing
# predictions = model.predict(X_test)

# to calc the accuracy, compare the predictions^ with the ACTUAL values we have in our output
# set for testing

# score = accuracy_score(y_test, predictions)
# print(score)

pickle.dump(model, open('LoL.pkl','wb'))
# openIt = pickle.load(open('model.pkl','rb'))