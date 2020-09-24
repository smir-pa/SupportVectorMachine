
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# reading csv file and extracting class column to y.
print('\n###################-loaded data-###################\n')
data = pd.read_csv("res/bill_authentication.csv")
print(data)
X = data.drop('Class', axis=1)
y = data['Class']
# making testing and training samples.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# regularization parameter C.
C = 35.0

print('\n#####################-linear-#####################\n')
svc_linear = SVC(kernel='linear', C=C)
svc_linear.fit(X_train, y_train)
y_pred = svc_linear.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred), 2))
print("Precision:", round(metrics.precision_score(y_test, y_pred), 2))
print("Recall:", round(metrics.recall_score(y_test, y_pred), 2))
print("F1-measure:", round(metrics.f1_score(y_test, y_pred), 2))

print('\n###################-polynomial-###################\n')
svc_poly = SVC(kernel='poly', degree=3, coef0=1, C=C)
svc_poly.fit(X_train, y_train)

y_pred = svc_poly.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred), 2))
print("Precision:", round(metrics.precision_score(y_test, y_pred), 2))
print("Recall:", round(metrics.recall_score(y_test, y_pred), 2))
print("F1-measure:", round(metrics.f1_score(y_test, y_pred), 2))

print('\n###################-Gaussian-###################\n')
svc_rbf = SVC(kernel='rbf', C=C, gamma='auto')
svc_rbf.fit(X_train, y_train)

y_pred = svc_rbf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred), 2))
print("Precision:", round(metrics.precision_score(y_test, y_pred), 2))
print("Recall:", round(metrics.recall_score(y_test, y_pred), 2))
print("F1-measure:", round(metrics.f1_score(y_test, y_pred), 2))

print('\n###################-sigmoid-###################\n')
svc_sigm = SVC(kernel='sigmoid', C=C, gamma=15)
svc_sigm.fit(X_train, y_train)

y_pred = svc_sigm.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred), 2))
print("Precision:", round(metrics.precision_score(y_test, y_pred), 2))
print("Recall:", round(metrics.recall_score(y_test, y_pred), 2))
print("F1-measure:", round(metrics.f1_score(y_test, y_pred), 2))

