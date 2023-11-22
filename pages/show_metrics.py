import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

train_data = pd.read_csv("data/train_data_cleaned.csv")
test_data = pd.read_csv("data/test_data_cleaned.csv")

X_train = train_data.iloc[:, :8]
y_train = train_data.iloc[:, 8:]

X_test = test_data.iloc[:, :8]
y_test = test_data.iloc[:, 8:]


rf_clf = pickle.load(open("models/rf.pkl", "rb"))
gnb_clf = pickle.load(open("models/gnb.pkl", "rb"))
logreg_clf = pickle.load(open("models/logreg.pkl", "rb"))


st.write("<h1>Metrics for models used for inference</h1>", unsafe_allow_html=True)


def show_metrics(model, model_name):
    st.write(f"""<h2>{model_name}</h2>""", unsafe_allow_html=True)
    st.write(f"Accuracy : {accuracy_score(y_test,model.predict(X_test))}")
    st.write(f"F1 score : {f1_score(y_test,model.predict(X_test),average='macro')}\n")


show_metrics(rf_clf, "Random Forest")
show_metrics(gnb_clf, "Gaussian Naive Bayes")
show_metrics(logreg_clf, "Logistic Regression")
