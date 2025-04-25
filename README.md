# 🌆 AI City Scout: The Brain of a Smart City During Crisis
  
### 🏆 First Prize Winner - Aggie Hackathon 2025

**Code:** [AI City Scout Python Code](https://github.com/YuwenAprilYang/Projects/blob/e165f48097d6a0c21692e55a89084e25b9a965e1/AI%20City%20Scout/app.py)  
**Report:** [AI City Scout Report](https://github.com/YuwenAprilYang/Projects/blob/b7e11fb1c320ab7ea49785cd6cf2bc56b0d2d94e/AI%20City%20Scout/AI%20City%20Scout%20Report.pdf)  
**Team:** Yuwen (April) Yang, Wenjun Song, Xinxin Liu  
**Tools:** Python, Streamlit, Random Forest, Isolation Forest, OpenAI GPT-4, Scikit-Learn, GeoPandas  

## Executive Summary
As natural disasters become more frequent and social misinformation spreads faster than emergency response systems can react, cities are left vulnerable.  
  
**_AI City Scout_** is a real-time, AI-powered disaster intelligence dashboard designed to predict disasters, flag fake social media posts, and generate actionable emergency plans — **all within seconds.**  
  
This project was awarded **First Prize (Innovator's Summit Award)** at the 2025 Aggie Hackathon for innovation, intelligence modeling and impact potential.

## Business Problem
In crisis situations like floods or earthquakes, city leaders face overwhelming volumes of noisy data and misinformation. Key questions we address:  
> - What type of disaster is unfolding right now, and where?  
> - Is social media reporting false information?  
> - How severe are the consequences?  
> - What immediate actions should the city take?

## System Overview
AI City Scout contains **three insteractive modules:**  
  
**1. Future Predictions**  
Uses real-time sensor + weather data to detect and predict disasters; Assesses hospital overload risk and severity; Generates AI-intergrated strategic plan
  
**2. Social Media Detector**  
Verifies tweet claiming a disaster by cross-checking sensor data near the time and location of the tweet
  
**3. Past Disasters Explorer**  
Interactive map showing historical disasters (2010-2015), allowing users to understand patterns and cascading risks

## Data Preprocessing
Dataset: 
- 50,000+ rows across sensors, weather, social media, infrastructure maps  
  
Preprocessing:  
- Outlier detection in weather via Isolation Forest
- Sensor classification via rule-based logic
- Location matching using geopy and GeoJSON infrastructure maps
- Data fusion of time-series, spatial, and text features  
  
## Core Models & Methods
### 1. Anomaly Detection
- Model: `Isolation Forest`
- Purpose: Detect outliers in sensor readings that may indicate disaster onset
- Example: Sudden spike in seismic activity or flood levels

### 2. Disaster Classification
- Uses sensor readings + weather data to classify disasters
- Manual rule-based classification refined with thresholds and weather cues (e.g., high wind speed and low pressure = Hurricane)

### 3. Severity Scoring
- Scale disaster intensity from 1 to 9 based on
    - Sensor reading value
    - Supporting weather factors (e.g., humidity, precipitation)

### 4. Hospital Overload Prediction
- Model: `Random Forest Classifier`
- Training Data: Historical disaster data (2010-2015)
- Target: Whether a disaster caused casualities > 10 or economic loss > 75th percentile
- Key Features:
    - Disaster type
    - Severrity (1-9)
    - Proximity to hospital

### 5. Fake Tweet Detection
- Method:
  - Checks tweet for keywords like "flood", "earthquake", etc.
  - Cross-validates location and time with sensor readings
- Flags tweet as "Real" or "Fake" accordingly

### 6. Emergency Response Generator
- Model: `OpenAI GPT-4`
- Based on the inputs (disaster type, severity level, hospital overload risk), generate actionable response plan
  
## Key Insights & Impact
### 1. Real-Time Disaster Prediction and Risk Scoring
- Predicts disasters before impact
- Applies a calibrate severity scale (1-9)  across disaster types (flood, fire, earthquake, hurricane)
- Assesses hospital overload risk to guide emergency priorities
> 📌 **Business takeaway:** Empowers cities to allocate resources proactively and mitigate cascading risks before they escalate  

### 2. Social Media Verification Reduces Panic
- Detects fake or misleading disaster tweets with **>90% accuracy**
- Validates tweet claims by checking for sensor evidence
> 📌 **Business takeaway:** Prevents unnecessary evacuations and builds public trust through reliable information screening  

### 3. Stategic Planning With AI-Generated Emergency Actions
- Each disaster instance is evaluated for severity and hospital risk, then passed to GPT-4 for action planning
- Delivers clear, city-level response plans **in seconds**
> 📌 **Business takeaway:** Accelerates emergency decision-making by **99.7%**, enabling faster, more coordinated response efforts

## Demonstation
_1. Predict future disasters from user's time input, showing possible results (severity & hospital overload risk), generating AI-integrated strategic plans_  

https://github.com/user-attachments/assets/ee36f817-154d-48cc-bb60-cb39abcac1dc


https://github.com/user-attachments/assets/c35cedfc-f4ef-4eff-8fb1-c628a2e247f9
  
_2. Flags fake disaster-related tweets by validating against physical evidence (user's input of time, location and text)_  

https://github.com/user-attachments/assets/1fe9310c-deb2-400c-bdbb-9433a5a42c1a
  
_3. Aggregates and visualizes historical events to uncover patterns_  

https://github.com/user-attachments/assets/1e68c290-2e1d-4822-8c8f-87ea2717fdc2


### 📊 Check out the detailed report and code to explore more insights!
