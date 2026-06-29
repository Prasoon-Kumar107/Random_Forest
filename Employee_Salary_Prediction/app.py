import streamlit as st
import pickle as p
import pandas as pd
st.title("Employee Salary Predictor")
st.markdown("### This webpage predicts wheather a employee earns more than 50k or not.")
st.sidebar.title("📌 Project Information")
st.sidebar.markdown("""
### Technologies Used
- 🤖 Machine Learning
- 📊 Scikit-learn
- 🎯 Random Forest Classifier
- ⚡ Streamlit
- GridSearchCV
""")
st.sidebar.markdown("""
### Model Details
    Algorithm: Random Forest Classifier
    Training Samples: 26048
    Categories: 2
    Accuracy: 86.0%
    ROC-AUC: 91.0%
""")
preprocessing = p.load(open("preprocessor.pkl","rb"))
model = p.load(open("model.pkl","rb"))
col1, col2 = st.columns(2)
with col1:
    age = st.slider("What is the age of Employee?",18,100,35)
    workclass = st.radio("What is the workclass of Employee?",['?','Private','State-gov','Federal-gov','Self-emp-not-inc',  'Self-emp-inc','Local-gov','Without-pay','Never-worked'],horizontal=True)
    education = st.radio("What is education level of Employee?",['HS-grad','Some-college','7th-8th','10th','Doctorate','Prof-school','Bachelors','Masters','11th','Assoc-acdm','Assoc-voc','1st-4th','5th-6th','12th','9th','Preschool'],horizontal=True)
    education_num = st.radio("What is education level of Employee expressed in numbers?",[ 9, 10,  4,  6, 16, 15, 13, 14,  7, 12, 11,  2,  3,  8,  5,  1],horizontal=True)
    marital_status = st.radio("What is the marital status of Employee?",['Widowed','Divorced','Separated','Never-married','Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'],horizontal=True)
    occupation = st.radio("What is the type of occupation of Employee?",['?','Exec-managerial','Machine-op-inspct','Prof-specialty','Other-service','Adm-clerical','Craft-repair','Transport-moving', 'Handlers-cleaners','Sales','Farming-fishing','Tech-support','Protective-serv','Armed-Forces','Priv-house-serv'],horizontal=True)
    relationship = st.radio("What is the relationship status of Employee?",['Not-in-family', 'Unmarried', 'Own-child' 'Other-relative', 'Husband','Wife'],horizontal=True)
with col2:
    race = st.radio("What is the race of Employee?",['White','Black','Asian-Pac-Islander','Other', 'Amer-Indian-Eskimo'],horizontal=True)
    sex = st.radio("What is sex of Employee?",['Female','Male'],horizontal=True)
    capital_gain = st.slider("What is the capital gain of Employee?",0,1000000,1080)
    capital_loss =  st.slider("What is the capital loss of Employee?",0,5000,90)
    hours_per_week = st.slider("How much hours does employee work per week?",1,100,40)
    native_country = st.radio("What is the native country of Employee?",['United-States','?','Mexico','Greece','Vietnam','China','Taiwan','India','Philippines','Trinadad&Tobago','Canada','South','Holand-Netherlands','Puerto-Rico','Poland','Iran','England','Germany','Italy','Japan','Hong','Honduras','Cuba','Irelan','Cambodia','Peru','Nicaragua','Dominican-Republic','Haiti','El-Salvador','Hungary','Columbia','Guatemala','Jamaica','Ecuador','France','Yugoslavia','Scotland','Portugal','Laos','Thailand','Outlying-US(Guam-USVI-etc)'],horizontal=True)
input_data = {
  'age' : age,
  'workclass' : workclass,
  'education' : education,
  'education.num' : education_num,
  'marital.status' : marital_status,
  'occupation' : occupation,
  'relationship' : relationship,
  'race' : race,
  'sex' : sex,
  'capital.gain' : capital_gain,
  'capital.loss' : capital_loss,
  'hours.per.week' : hours_per_week,
  'native.country' : native_country,
}
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)
if st.button("Predict"):
    if prediction == 0:
        st.success("The Employee's salary is likely to be below 50K.")
    else:
        st.success("The Employee'ssalary is likely to be above 50K.")
    probabilities = model.predict_proba(input_df)[0]
    confidence = round(probabilities.max(),2)*100
    st.success(f"Confidence Score for Prediction is {confidence}%")
else:
    st.info("Please choose from all the above options above and then press Predict.")
st.image("Feature_importance.png","Most important features for prediction",use_container_width=True)