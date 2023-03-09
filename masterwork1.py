
#pip install streamlit


import numpy as np
import streamlit as st
import pickle
from pickle import dump,load


st.header('Heart diseases prediction')

"""
- **height**    144 - 200
- **weight**    45-180
-  **ap_hi**    90 - 180
-  **ap_lo**    60 - 120

"""

def load ():
    with open("model2.pcl", "rb") as fid:
      return pickle.load(fid)

model = load()
#(['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
#       'smoke', 'alco', 'active', 'age_year'
# в X_test_c забыл поменять тип 'weight' на int

gender = st.selectbox('Пол', [1,2])
gluc = st.selectbox('Gluc', [1,2,3])
cholesterol = st.selectbox('Cholesterol', [1,2,3])
smoke = st.checkbox('Вы курите?')
alco = st.checkbox('Вы употребляете алкоголь?')
active = st.checkbox('Вы заниметесь спортом?')
age_year = st.slider('Возраст', 20, 80)
height = st.slider('Рост', 144, 200)
weight = st.slider('Вес', 45, 180)
ap_hi = st.slider('Верхнее Давление ', 90, 180)
ap_lo = st.slider('Нижнее Давление ', 60, 120)

y_pr = model.predict_proba([[gender, height, weight, ap_hi, ap_lo,cholesterol, gluc, smoke,alco,active, age_year]])[:,1]
# Так и не понял почему значение переменной y_pr выводится с новой строки
st.write('Вероятность появления сердечных заболеваний, %', np.round((y_pr*100),2))
