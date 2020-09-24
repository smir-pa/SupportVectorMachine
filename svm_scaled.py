
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

# reading csv file and extracting class column to y.
print('\n###################-loaded data-###################\n')
data = pd.read_csv("res/bill_authentication.csv")
print(data)
X = data.drop('Class', axis=1)
y = data['Class']

# scaling data matrix to the [0, 1] range
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)
print("Scaled data:\n", X_minmax)

# making testing and training samples.
X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.2)
# regularization parameter C.
C = 35.0

print('\n#####################-linear-#####################\n')
svc_linear = SVC(kernel='linear', C=C)
svc_linear.fit(X_train, y_train)

y_pred = svc_linear.predict(X_test)

print("Number of support vectors for each class:", svc_linear.n_support_)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred), 2))
print("Precision:", round(metrics.precision_score(y_test, y_pred), 2))
print("Recall:", round(metrics.recall_score(y_test, y_pred), 2))
print("F1-measure:", round(metrics.f1_score(y_test, y_pred), 2))

print('\n###################-polynomial-###################\n')
svc_poly = SVC(kernel='poly', degree=3, coef0=1, C=C)
svc_poly.fit(X_train, y_train)

y_pred = svc_poly.predict(X_test)

print("Number of support vectors for each class:", svc_poly.n_support_)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred), 2))
print("Precision:", round(metrics.precision_score(y_test, y_pred), 2))
print("Recall:", round(metrics.recall_score(y_test, y_pred), 2))
print("F1-measure:", round(metrics.f1_score(y_test, y_pred), 2))

print('\n###################-Gaussian-###################\n')
svc_rbf = SVC(kernel='rbf', C=C, gamma='auto')
svc_rbf.fit(X_train, y_train)

y_pred = svc_rbf.predict(X_test)

print("Number of support vectors for each class:", svc_rbf.n_support_)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred), 2))
print("Precision:", round(metrics.precision_score(y_test, y_pred), 2))
print("Recall:", round(metrics.recall_score(y_test, y_pred), 2))
print("F1-measure:", round(metrics.f1_score(y_test, y_pred), 2))

print('\n###################-sigmoid-###################\n')
svc_sigm = SVC(kernel='sigmoid', C=C, gamma='auto')
svc_sigm.fit(X_train, y_train)

y_pred = svc_sigm.predict(X_test)

print("Number of support vectors for each class:", svc_sigm.n_support_)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred), 2))
print("Precision:", round(metrics.precision_score(y_test, y_pred), 2))
print("Recall:", round(metrics.recall_score(y_test, y_pred), 2))
print("F1-measure:", round(metrics.f1_score(y_test, y_pred), 2))
# print("Never predicted label:", set(y_test) - set(y_pred))


# tSNE
# fitting and transforming data with a TSNE
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)
target_ids = range(2)
# visualizing the data
plt.figure(figsize=(6, 5))
colors = 'purple', 'orange'
for i, c, label in zip(target_ids, colors, ['0', '1']):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
plt.legend()
plt.show()


# making four graphs based on two features
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# Take the first two features. We could avoid this by using a two-dim dataset
a = np.array(X_minmax)
b = np.array(data)
X = a[:, :2]
y = b[:, 4]

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
models = (svc_linear,
          svc_poly,
          svc_rbf,
          svc_sigm)

models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'SVC with polynomial kernel)',
          'SVC with RBF kernel',
          'SVC with sigmoid kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Variance')
    ax.set_ylabel('Skewness')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
