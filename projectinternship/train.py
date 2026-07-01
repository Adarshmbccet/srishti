import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv("train.csv")

print(df.head())

print(df.shape)
print(df.info())

print(df.describe())
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop("Cabin", axis=1)
df = df.drop(
    ["PassengerId", "Name", "Ticket"],
    axis=1
)
print(df.head())
le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])
print(df.head())
X = df.drop("Survived", axis=1)

y = df["Survived"]
print(X.shape)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(accuracy)

print(classification_report(
    y_test,
    y_pred
))
os.makedirs(
    "models",
    exist_ok=True
)
joblib.dump(
    model,
    "models/titanic_model.joblib"
)
model_info = {

    "model_type":"Random Forest",

    "accuracy":float(accuracy),

    "feature_names":X.columns.tolist(),

    "target_names":[
        "Did Not Survive",
        "Survived"
    ]

}
with open(
    "models/model_info.json",
    "w"
) as f:

    json.dump(
        model_info,
        f,
        indent=4
    )
feature_ranges = {

"Pclass":{

"min":1,

"max":3,

"default":2

},

"Age":{

"min":0,

"max":80,

"default":30

},

"SibSp":{

"min":0,

"max":8,

"default":0

},

"Parch":{

"min":0,

"max":6,

"default":0

},

"Fare":{

"min":0,

"max":600,

"default":50

}

}
with open(
"models/feature_ranges.json",
"w"
) as f:

    json.dump(
        feature_ranges,
        f,
        indent=4
    )

