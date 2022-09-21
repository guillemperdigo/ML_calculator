#!/usr/bin/env python
# coding: utf-8

# In[15]:


import streamlit as st
import sklearn

# load model
import pickle
loaded_model = pickle.load(open('models/knn_calculator.sav', 'rb'))

model_name = str(loaded_model.get_params()["estimator"])
param_combinations = str(loaded_model.get_params()["n_iter"])

string = "A " + model_name + " model was trained with " + param_combinations + " combinations of parameters to learn how to sum. Try it out!"



st.title("Machine Learning Calculator")

st.write(string)



# new house with fake data
import pandas as pd
input1 = st.number_input("Number 1")
input2 = st.number_input("Number 2")

input_df = pd.DataFrame({"n1":[input1], "n2":[input2]})

# prediction
st.write("The result of your sum as calculated by Machine Learning is: ", loaded_model.predict(input_df))
