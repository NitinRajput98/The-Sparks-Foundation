import numpy as np
import pickle
import pandas as pd
import streamlit as st
import sklearn.linear_model
from PIL import Image

pickle_in = open("linear.pkl", "rb")
model = pickle.load(pickle_in)

def predict_score(hours):
    prediction = model.predict(np.array([[hours]], dtype='float64'))
    print(prediction.flatten())
    return prediction.flatten()

@st.cache(persist= True)
def load_data():
    data = pd.read_csv("student_scores - student_scores.csv")
    return data

def main():
    st.title('Scores Prediction App')
    st.sidebar.title('Scores Prediction App')
    st.sidebar.subheader('Description')
    st.sidebar.markdown('This regression app will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.')
    st.markdown("How much score  can be expected based upon the number of hours studied? 📝 📚")

    hours = st.text_input('Number of hours student studies')
    result = ''

    if st.button("Predict"):
        result = predict_score(hours)
        result = result[0].round(2)
    st.success('Score predicted is : {}'.format(result))

    df = load_data()
    if st.sidebar.checkbox('Show raw data',False):
        st.subheader('Student Scores Dataset(Regression)')
        st.write(df)
        st.markdown('This dataset contains only two columns that are the scores achieved by the student and the number of hours a student studies.')

    if st.sidebar.checkbox('Show accuracy of model',False):
        st.subheader('Accuracy of the model')
        st.write('Mean Absolute Error: 4.18')

    if st.sidebar.checkbox('Show Plot',False):
        st.subheader('Best fitted line')
        image = Image.open('plot.jpg')
        st.image(image,caption='Scatter Plot with fitted line',use_column_width=True)



if __name__ == '__main__':
    main()