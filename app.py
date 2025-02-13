import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime

def load_data():
    data = pd.read_csv('./data/historical_air_quality_2021_en.csv')
    data = data[data['Station name'].str.contains('Ho Chi Minh City', na=False)]
    
    # convert to datetime
    data['Date'] = pd.to_datetime(data['Data Time S'], errors='coerce')
    
    # convert to numeric
    data['AQI Value'] = pd.to_numeric(data['AQI index'], errors='coerce')
    
    # remove error data
    data = data.dropna(subset=['Date', 'AQI Value'])
    return data

def train_model(data):
    # convert date -> ordinal
    data['Date_ordinal'] = data['Date'].map(datetime.toordinal)
    X = data[['Date_ordinal']]
    y = data['AQI Value']
    
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, 'aqi_predictor.pkl')
    return model

def predict_aqi(model, dates):
    dates_ordinal = np.array([datetime.toordinal(d) for d in dates]).reshape(-1, 1)
    predictions = model.predict(dates_ordinal)
    
    # value must >= 0
    predictions = np.maximum(predictions, 0)
    return predictions

def show_city_stats(data):
    st.subheader("Thống kê ô nhiễm - TP.HCM")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AQI trung bình", f"{data['AQI Value'].mean():.1f}")
    with col2:
        st.metric("AQI cao nhất", f"{data['AQI Value'].max():.1f}")
    with col3:
        st.metric("AQI thấp nhất", f"{data['AQI Value'].min():.1f}")

    # AQI tendency
    st.subheader("Diễn biến AQI theo thời gian")
    fig, ax = plt.subplots(figsize=(10, 4))
    data.plot(x='Date', y='AQI Value', ax=ax)
    st.pyplot(fig)

def main():
    st.title("Dự đoán AQI cho TP.HCM")
    data = load_data()
    
    st.subheader("Dữ liệu AQI lịch sử")
    st.line_chart(data.set_index('Date')['AQI Value'])

    if st.button("Huấn luyện mô hình dự đoán AQI"):
        model = train_model(data)
        st.success("Mô hình đã được huấn luyện thành công!")
    
    # predict AQI
    st.subheader("Dự đoán AQI trong tương lai")
    future_dates = st.text_area("Nhập các ngày cần dự đoán (YYYY-MM-DD, mỗi dòng một ngày):", 
                                "2026-01-01\n2027-01-01\n2027-12-31")
    
    if st.button("Dự đoán"):
        try:
            model = joblib.load('aqi_predictor.pkl')
            dates_to_predict = [datetime.strptime(d.strip(), "%Y-%m-%d") for d in future_dates.split('\n')]
            
            if any(date.year > 2025 for date in dates_to_predict):
                st.error("Chỉ cho phép dự đoán đến năm 2025. Vui lòng nhập ngày hợp lệ.")
                return
            
            predictions = predict_aqi(model, dates_to_predict)
            
            st.subheader("Kết quả dự đoán")
            for date, aqi in zip(dates_to_predict, predictions):
                st.write(f"{date.strftime('%Y-%m-%d')}: AQI dự đoán = {aqi:.1f}")
                
                if date.year > 2022:
                    st.warning(f"Dự đoán cho ngày {date.strftime('%Y-%m-%d')} vượt quá phạm vi dữ liệu. Kết quả có thể không chính xác.")
        
        except FileNotFoundError:
            st.error("Vui lòng huấn luyện mô hình trước!")
        except ValueError as e:
            st.error(f"Lỗi định dạng ngày tháng: {str(e)}")
            
    # show stats
    show_city_stats(data)

if __name__ == "__main__":
    main()
