from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,auc
import pandas as pd
import joblib
import argparse

data=pd.read_csv("./diabetes.csv")
print(data.columns)

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X, y, random_state=101, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))

y_pred=rf.predict(X_test)
classification_report(y_test,y_pred)
confusion_matrix(y_test,y_pred)

acc=accuracy_score(y_test,y_pred)
print("Accuracy",acc)

print(accuracy_score(y_test,y_pred))
joblib.dump(rf, "diabetis_rf.pkl")