import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("train.csv")

features = ["Pclass","Sex","Age","Fare"]
X = data[features]
y = data["Survived"]

#handle missing values
X = X.copy()
X["Age"] = X["Age"].fillna(X["Age"].mean())
#convert categorial to numeric
X["Sex"] = X["Sex"].map({"male": 0,"female":1})

#split data
X_train, X_test, y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#train model
model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

#test model
y_pred = model.predict(X_test)
print("accuracy:",accuracy_score(y_test,y_pred))

test_data = pd.read_csv("test.csv")
# Make a copy to avoid warnings
test_X = test_data.copy()

# Fill missing Age values
test_X["Age"] = test_X["Age"].fillna(test_X["Age"].mean())

# Fill missing Fare values (if any)
test_X["Fare"] = test_X["Fare"].fillna(test_X["Fare"].mean())

# Convert Sex to numeric
test_X["Sex"] = test_X["Sex"].map({"male": 0, "female": 1})

test_X = test_X[features]  # features = ["Pclass","Sex","Age","Fare"]

predictions = model.predict(test_X)

output = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})

output.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")
