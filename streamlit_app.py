import pandas as pd
import streamlit as st
import joblib
import numpy as np

def main():
    model = joblib.load("models/model.pkl")

    st.title('Linear Regression Model')

    feature = st.sidebar.number_input("Input", value=0.0)

    user_input = pd.DataFrame({'Input': [feature]})

    if st.sidebar.button("Predict"):

        prediction = model.predict(user_input)

        st.sidebar.success(f"Predicted Output: {prediction}")

    st.subheader("User Input:")
    st.write(user_input)

if __name__ == "__main__":
    main()



