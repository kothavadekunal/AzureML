from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score,recall_score
import pandas as pd
import joblib
import os
import argparse

from azureml.core import Run
run = Run.get_context()

data=pd.read_csv("./diabetes.csv")
print(data.columns)

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X, y, random_state=101, test_size=0.2)

rf = RandomForestClassifier(n_estimators = 500,random_state=0)
rf.fit(X_train, y_train)

run.log('Parameters',rf.get_params())

print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))

y_pred=rf.predict(X_test)

classification_report(y_test,y_pred)
confusion_matrix(y_test,y_pred)

acc=accuracy_score(y_test,y_pred)
print("Accuracy",acc)
run.log('Accuracy',acc)

precision = precision_score(y_test,y_pred)
print("Precision",precision)
run.log("Precision",precision)

recall = recall_score(y_test,y_pred)
print("Recall",recall)
run.log("Recall",recall)

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value = rf, filename = "outputs/diabetes_rf.pkl")

run.complete()