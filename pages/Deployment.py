import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("models/gnb.pkl", "rb"))


def load_model(model_choice):
    if model_choice == "Logistic Regression":
        return pickle.load(open("models/logreg.pkl", "rb"))
    if model_choice == "Naive Bayes":
        return pickle.load(open("pages/models/gnb.pkl", "rb"))
    # if model_choice == 'XGBoost':
    #     return pickle.load(open('models\xgb.pkl', 'rb'))
    if model_choice == "Random Forest":
        return pickle.load(open("models/rf.pkl", "rb"))


def suggest_crop(N, P, K, temperature, humidity, ph, rainfall, model_choice):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]]).astype(
        np.float32
    )
    # model = load_model(model_choice)
    predictions = model.predict(features)
    # probabilities = model.predict_proba(features)
    return predictions


def main():
    st.title("Crop Consultant")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Crop Suggestion using soil and climate </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    models = [
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest",
    ]
    model_choice = st.radio("Model choice", models)
    N = st.number_input("Nitrogen content")
    P = st.number_input("Phosphorus content")
    K = st.number_input("Potassium content")
    temperature = st.number_input(
        "Temperature in Celsius", min_value=15.0, max_value=40.0
    )
    humidity = st.number_input("Humidity")
    ph = st.number_input("PH Value", min_value=3.5, max_value=10.0)
    rainfall = st.number_input("Rainfall in mm")

    if model_choice is not None:
        if st.button("Suggest crop to grow"):
            output = suggest_crop(
                N, P, K, temperature, humidity, ph, rainfall, model_choice
            )

            st.success(f"Crop to be grown is {output[0]}")
            # st.write(proba[:, 1])
    else:
        st.warning("Model choice cannot be none")


if __name__ == "__main__":
    main()
