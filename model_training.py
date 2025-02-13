# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# import joblib
# from datetime import datetime

# # Train prediction model
# @st.cache_data # Cache the model training
# def train_model(data):
#     data.dropna(subset=['Date', 'AQI Value'], inplace=True) # Handle missing data
#     data['Date_ordinal'] = data['Date'].map(datetime.toordinal)
#     X = data[['Date_ordinal']]
#     y = data['AQI Value']
    
#     model = LinearRegression()
#     model.fit(X, y)
#     joblib.dump(model, 'aqi_predictor.pkl')
#     return model

# # Make predictions
# def predict_aqi(model, dates):
#     dates_ordinal = np.array([datetime.toordinal(d) for d in dates]).reshape(-1, 1)
#     return model.predict(dates_ordinal)