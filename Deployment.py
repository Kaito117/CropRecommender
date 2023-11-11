import streamlit as st
import pickle
import numpy as np

model = pickle.load("GaussianNB().pkl", "rb")

def load_model(model_choice):
    if model_choice =='Logistic Regression':
        return pickle.load(open('models\logreg.pkl','rb'))
    if model_choice == 'Support Vector Classifier':
        return pickle.load(open('models\svm.pkl', 'rb'))
    if model_choice == 'Naive Bayes':
        return pickle.load(open('models\gnb.pkl', 'rb'))
    if model_choice == 'XGBoost':
        return pickle.load(open('models\xgb.pkl', 'rb'))
    if model_choice == 'Random Forest':
        return pickle.load(open('models\svm.pkl', 'rb'))


def suggest_crop(N, P, K, temperature, humidity, ph, rainfall,model_choice):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]]).astype(
        np.float32
    )
    model=load_model(model_choice)
    predictions = model.predict(features)
    return predictions


def main():
    st.title("Crop Consultant")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Crop Suggestion using soil and climate </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    models=['Logistic Regression','Support Vector Classifier','Naive Bayes','XGBoost','Random Forest']
    model_choice = st.radio('Model choice',models)

    if model_choice is not None:
        N = st.number_input("Nitrogen content")
        P = st.number_input("Phosphorus content")
        K = st.number_input("Potassium content")
        temperature = st.number_input("Temperature in Celsius",min_value=15, max_value=40)
        humidity = st.number_input("Humidity")
        ph = st.number_input("PH Value",min_value=3.5,max_value=10)
        rainfall = st.number_input("Rainfall in mm")

        if st.button("Suggest crop to grow"):
            output = suggest_crop(N, P, K, temperature, humidity, ph, rainfall,model_choice)

        st.success(f"Crop to be grown is {output}")

        st.write(output)
    else:
        st.warning('Model choice cannot be none')
