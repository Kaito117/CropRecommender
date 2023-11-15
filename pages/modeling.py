import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import warnings

warnings.filterwarnings("ignore")


seed = 41


train_data = pd.read_csv("train_data_cleaned.csv")
test_data = pd.read_csv("test_data_cleaned.csv")

X_train = train_data.iloc[:, 1:8]
y_train = train_data.iloc[:, 8:]

X_test = test_data.iloc[:, 1:8]
y_test = test_data.iloc[:, 8:]


def print_metrics(classifier, clf_name, classification_report=False):
    print(clf_name)
    print(f"Accuracy : {accuracy_score(y_test,classifier.predict(X_test))}")
    print(f1_score(y_test, classifier.predict(X_test), average="macro"))
    print()
    if classification_report:
        print(classification_report(y_test, classifier.predict(X_test)))


logreg_clf = LogisticRegression(random_state=seed)
logreg_clf.fit(X_train, y_train)
print_metrics(logreg_clf, "Logistic Regression")


svc_clf = SVC(random_state=seed)
svc_clf.fit(X_train, y_train)
print_metrics(svc_clf, "Support Vector machine")


gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)
print_metrics(gnb_clf, "Gaussian Naive Bayes")


rf_clf = RandomForestClassifier(random_state=seed)
rf_clf.fit(X_train, y_train)
print_metrics(rf_clf, "Random Forest")

models = [logreg_clf, gnb_clf, svc_clf, rf_clf]
filenames = ["logreg", "gnb", "svc", "rf"]


def store_models(store=False):
    if store:
        for idx, filename in enumerate(filenames):
            with open(filename + ".pkl", "wb") as file:
                pickle.dump(models[idx], file)


# store_models(True)
