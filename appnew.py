import streamlit as st
import joblib
import numpy as np
import pandas as pd

preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")


def main():
    st.title("Model Prediksi Serangan Jantung")

    age = st.number_input('Masukkan Umur', min_value=17, max_value=100, value=1)
    trestbps = st.number_input("Tekanan Darah (restbp)", min_value=90, max_value=200, value=1)
    chol = st.number_input("Kolestrol", 100, 600, 50)
    thalach = st.number_input("Tekanan Darah Maksimal (thalach)", 100, 200, 10)
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0)
    sex = st.selectbox("Jenis Kelamin (0 = Wanita, 1 = Pria)", [0, 1])
    cp = st.selectbox("Sakit Dada", [0, 1, 2, 3])
    fbs = st.selectbox("Tingkat gula darah > 120 mg/dl (0 = Tidak, 1 = Ya)", [0, 1])
    restecg = st.selectbox("Hasil elektrodiagram istirahat (estecg)", [0, 1, 2])
    exang = st.selectbox("Angina olahraga (0 = Tidak, 1 = Ya)", [0, 1])
    slope = st.selectbox("Slope saat puncak olahraga (slope)", [0, 1, 2])
    ca = st.selectbox("Jumlah pembuluh utama yang terlihat (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    if st.button("Make Prediction"):
        result = make_prediction(age, sex, cp, trestbps, chol, fbs,
                                 restecg, thalach, exang, oldpeak,slope, ca, thal)

        if result == 1:
            st.error("Resiko tinggi serangan jantung")
        else:
            st.success("Resiko rendah serangan jantung")


def make_prediction(age, sex, cp, trestbps, chol, fbs,
                    restecg, thalach, exang, oldpeak,slope, ca, thal):

    input_dict = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    input_df = pd.DataFrame([input_dict])

    X_processed = preprocessor.transform(input_df)
    prediction = model.predict(X_processed)

    return prediction[0]


if __name__ == "__main__":
    main()
