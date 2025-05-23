import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import pickle

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv(r'C:\Users\hp\Documents\ml1\data\heart.csv')


# Data Preprocessing
# Encode categorical variables


label_encoders = {}
for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

with open(r'C:\Users\hp\Documents\ml1\models\encoders.pickle', 'wb') as handle:
    pickle.dump(label_encoders, handle, protocol=pickle.HIGHEST_PROTOCOL)




# Feature Selection
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

with open(r'C:\Users\hp\Documents\ml1\models\scaler.pickle', 'wb') as handle:
    pickle.dump(scaler,handle)


# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

with open(r'C:\Users\hp\Documents\ml1\models\model.pickle', 'wb') as handle:
    pickle.dump(model,handle)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))


