import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler , LabelEncoder

#we will now open file, load the model:

def load_model(): #this is function to load model
    with open('student_lr_final_model.pkl', 'rb') as file:
 # we are reading the file
       model , scaler , le =  pickle.load(file) #we r loading the file and we know that model /file will give 3 outputs so we are storing it into 3 variables
    
    return model , scaler , le

# now we will transform the data which user will give as per our model, tht ie label encoder for extra activity and scaler / std normal dis for entire data

def pre_processing_inputdata (data , scaler , le) : 
    data ['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    #above we will label encode the extra activity data point from data entered by user and store and update the value
    # datavariable is the data entered by user
    df = pd.DataFrame([data]) # we will store the data in the data frame
    #now we will transform the data into normal distribution

    df_transform = scaler.transform (df)
    return df_transform

#now we will predict the result

def predict_data(data):
    model, scaler, le = load_model()  #this will store return of load model fun into 3 variables
    processed_data = pre_processing_inputdata (data, scaler, le)  #this will transform the data & store it into processed data
    
    #now we will predict the data
    prediction = model.predict(processed_data)
    return prediction
    
#app building starts from here 

def main (): #this is my main function that will be called in streamlit
    st.title ('Student Performance Prediction')
    st.write ("Enter your data to predict the performance of the student - ")

    Hours_studied = st.number_input ('Hours studied', min_value= 1, max_value= 10, value= 5)
    Previous_score = st.number_input ('Previous score', min_value= 40, max_value= 100, value= 70)
    Extracurricular_Activities = st.selectbox ('Extracurricular Activities',['Yes', 'No'])
    Sleeping_Hours =st.number_input ('Sleeping Hours', min_value= 4, max_value= 10, value= 7)
    Ques_solved = st.number_input ('No of question paper solved', min_value= 0, max_value= 10, value= 5)

    #we will now do a data mapping

    if st.button ('Predict your score'):
        user_data = {
            'Hours Studied' : Hours_studied,
            'Previous Scores' : Previous_score,
            'Extracurricular Activities' : Extracurricular_Activities,
            'Sleep Hours' : Sleeping_Hours,
            'Sample Question Papers Practiced' : Ques_solved

        }
        prediction = predict_data(user_data)
        st.success(f'Your predicated score value is - {prediction}')


if __name__ == '__main__':
    main()
