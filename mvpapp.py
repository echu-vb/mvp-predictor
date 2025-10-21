#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 'vorp', 'ws_48', 'win%', 'l', 'per'


# In[1]:


import streamlit as st
import joblib
import numpy as np


# In[4]:


model = joblib.load('model.pkl')

st.title("NBA MVP Prediction")
st.divider()
st.write("This model takes input of the 5 most important features for determining MVPs to predict whether or not they won the MVP award!")
st.write("Disclaimer: Model may have trouble predicting MVPs from before the 90s")
st.write("As an example, you can have the model determine whether or not SGA should've won MVP in 2025 using the stats found here: https://www.basketball-reference.com/players/g/gilgesh01.html#all_advanced")
st.write("Value Over Replacement: a box score estimate of the points per 100 TEAM possessions that a player contributed above a replacement-level (-2.0) player, translated to an average team and prorated to an 82-game season.")
st.write("Win Shares Per 48 Minutes: an estimate of the number of wins contributed by the player per 48 minutes")
st.write("Win% = Regular Season Wins / (Regular Season Wins + Regular Season Losses)")
st.write("Player Efficiency Rating: sums up all a player's positive accomplishments, subtracts the negative accomplishments, and returns a per-minute rating of a player's performance.")
st.divider()
vorp = st.number_input('VORP', min_value=0.0)
ws48 = st.number_input('WS_48', min_value=0.0, format='%.3f')
winper = st.number_input('Win Percentage', min_value=0.0, max_value=1.0, format='%.3f')
l = st.number_input('Losses', min_value=0)
per = st.number_input('PER', min_value=0.0)
st.divider()

button = st.button('Predict')

if button:
    X = np.array([vorp, ws48, winper, l, per])
    X_array = X.reshape(1, -1)
    prediction = model.predict(X_array)[0]
    predicted = 'MVP' if prediction == True else 'Not MVP'
    st.write(predicted)
    st.balloons()
else:
    st.write('Enter values and then press the button!')


# In[ ]:




