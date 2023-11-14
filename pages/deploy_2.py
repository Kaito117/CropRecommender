import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("models/GaussianNB().pkl", "rb"))


def suggest_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]]).astype(
        np.float32
    )
    predictions = model.predict(features)
    return predictions


def main():
    st.title("Agro Consultant")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Crop suggestion using soil nutrients </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    N = st.number_input("Nitrogen content")
    P = st.number_input("Phosphorus content")
    K = st.number_input("Potassium content")
    temperature = st.number_input("Temperature")
    humidity = st.number_input("Humidity")
    ph = st.number_input("PH Value")
    rainfall = st.number_input("Rainfall in mm")

    if st.button("Suggest crop"):
        output = suggest_crop(N, P, K, temperature, humidity, ph, rainfall)

        st.success(f"Crop to be grown is {output[0]}")


if __name__ == "__main__":
    main()
