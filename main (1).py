import numpy as np
import pickle as pk
import streamlit as st
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
loaded_model = pk.load(open("trained_model.sav", "rb"))


def hpp(input_data):
    # input_data = (4.98, 2.31, 0.538, 15.3, 6.575, 296, 4.09, 65)
    arr = np.asarray(input_data)
    arr = arr.reshape(1, -1)
    ar = sc.fit_transform(arr)

    prediction = loaded_model.predict(ar)
    return (f"The predicted house value  {round(prediction[0], 2)}")


def main():
    # giving a title
    st.title("Housing Price Prediction")

    # getting the input data from user
    result = 0

    bedrooms = st.text_input("Number of bedroom in house")
    bathrooms = st.text_input("Number of bathroom in house")
    sqft_living = st.text_input("sqft_living")
    sqft_lot = st.text_input("Square footage of the land space")
    floors = st.text_input("Number of floors")
    # waterfront = st.text_input("waterfront or not")
    view = st.text_input(" view of the property")
    # condition = st.text_input(" condition of the apartment")
    grade = st.text_input("grade for property")
    # sqft_above = st.text_input("sqft above above ground level")
    sqft_basement = st.text_input("sqft_basement")
    yr_built = st.text_input("built year")
    yr_renovated = st.text_input("renovated or not")
    zipcode = st.text_input("zipcode area")
    lat = st.text_input("Lattitude")
    long = st.text_input("Longitude")
    sqft_living15 = st.text_input("living space for the nearest 15 neighbors")
    sqft_lot15 = st.text_input("land lots of the nearest 15 neighbors")
    year =  st.text_input("Price Year")
    month= st.text_input("Month")
    dates=st.text_input("Date")

    # code for prediction

    # creating a button for prediction
    if st.button("Predict"):
        result = hpp(
            [bedrooms, bathrooms, sqft_living, sqft_lot, floors, view, grade,
             sqft_basement,
             yr_built, yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15, year,month,dates])
    st.success(result)


if __name__ == "__main__":
    main()
