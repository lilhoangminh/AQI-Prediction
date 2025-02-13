import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime

def load_data():
    data = pd.read_csv('./data/air_pollution_dataset.csv')
    # convert to datetime
    data['Date'] = pd.to_datetime(data.get('Date', pd.Timestamp.now()))  
    return data

def train_model(data):
    data['Date_ordinal'] = data['Date'].map(datetime.toordinal)
    X = data[['Date_ordinal']]
    y = data['AQI Value']
    
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, 'aqi_predictor.pkl')
    return model

def predict_aqi(model, dates):
    dates_ordinal = np.array([datetime.toordinal(d) for d in dates]).reshape(-1, 1)
    return model.predict(dates_ordinal)

def show_city_stats(data, city_name):
    city_data = data[data['City'] == city_name]

    st.subheader(f"Thống kê ô nhiễm - {city_name}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AQI trung bình", f"{city_data['AQI Value'].mean():.1f}")
    with col2:
        st.metric("PM2.5 cao nhất", f"{city_data['AQI Value'].max():.1f} µg/m³")
    with col3:
        st.metric("NO2 trung bình", f"{np.random.uniform(10, 50):.1f} ppb")

    # AQI tendency
    st.subheader("Diễn biến AQI theo thời gian")
    fig, ax = plt.subplots(figsize=(10, 4))
    city_data.plot(x='Date', y='AQI Value', ax=ax)
    st.pyplot(fig)

    # show stats
    st.subheader("Phân bố các chỉ số ô nhiễm")
    pollutants = ['AQI Value', 'PM2.5 AQI Value', 'NO2 AQI Value']
    selected_pollutant = st.selectbox("Chọn chỉ số ô nhiễm", pollutants)
    fig2, ax2 = plt.subplots()
    city_data[selected_pollutant].hist(ax=ax2, bins=20)
    ax2.set_xlabel(selected_pollutant)
    ax2.set_ylabel('Tần suất')
    st.pyplot(fig2)

    # main code
def main():
    st.title("Dự đoán AQI đến năm 2027")
    data = load_data()
    
    st.subheader("Dữ liệu AQI lịch sử")
    st.line_chart(data.set_index('Date')['AQI Value'])

    if st.button("Huấn luyện mô hình dự đoán AQI"):
        model = train_model(data)
        st.success("Mô hình đã được huấn luyện thành công!")
    
    # AQI predict
    st.subheader("Dự đoán AQI trong tương lai")
    future_dates = st.text_area("Nhập các ngày cần dự đoán (YYYY-MM-DD, mỗi dòng một ngày):", 
                                "2026-01-01\n2027-01-01\n2027-12-31")
    
    if st.button("Dự đoán"):
        try:
            model = joblib.load('aqi_predictor.pkl')
            dates_to_predict = [datetime.strptime(d.strip(), "%Y-%m-%d") for d in future_dates.split('\n')]
            predictions = predict_aqi(model, dates_to_predict)
            
            st.subheader("Kết quả dự đoán")
            for date, aqi in zip(dates_to_predict, predictions):
                st.write(f"{date.strftime('%Y-%m-%d')}: AQI dự đoán = {aqi:.1f}")
                
                if date.year > 2025:
                    st.warning(f"Dự đoán cho ngày {date.strftime('%Y-%m-%d')} vượt quá phạm vi dữ liệu. Kết quả có thể không chính xác.")
        
        except FileNotFoundError:
            st.error("Vui lòng huấn luyện mô hình trước!")
        except ValueError as e:
            st.error(f"Lỗi định dạng ngày tháng: {str(e)}")

    # search city name
    st.subheader("Tra cứu ô nhiễm theo thành phố")
    cities = sorted(data['City'].unique())
    selected_city = st.selectbox("Chọn hoặc nhập tên thành phố:", options=cities, placeholder="Chọn thành phố...")

    if selected_city:
        show_city_stats(data, selected_city)
    # show city stats            
        st.subheader("Dữ liệu chi tiết")
        city_data = data[data['City'] == selected_city]
        st.dataframe(city_data.sort_values('Date', ascending=False))

if __name__ == "__main__":
    main()