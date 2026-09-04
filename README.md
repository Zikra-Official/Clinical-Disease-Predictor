# 🩺 Clinical Breast Cancer Risk Predictor

An end-to-end interactive diagnostic Web Application designed to predict breast cancer risk levels (Malignant vs. Benign) using Machine Learning and Streamlit 🚀.

---

## 📌 Project Overview
This clinical predictor system analyzes tumor measurement parameters to assess cancer risk. It features:
* 🎛️ **Interactive Diagnostics:** Dynamic sliders with real-time risk calculation.
* 🔴 **Quick Sample Data:** Presets to quickly test High Risk (Malignant) and Low Risk (Benign) cases.
* 🖼️ **Anatomical Visualization:** Simulated 2D shape profile of the tumor based on radius and smoothness.
* 📊 **Risk Reports:** Comprehensive text summaries and graphical probability charts.
* 🎨 **UI Customization:** Toggle between Light and Dark themes.

---

## 🛠️ Tech Stack & Libraries
* 🐍 **Language:** Python
* 🌐 **Web Framework:** Streamlit
* 🤖 **Machine Learning:** Scikit-Learn, Joblib
* 📊 **Data Visualization:** Plotly Graph Objects, Pandas, NumPy

---

## 🏗️ Core Features
1. 🩺 **Risk Assessment:** Estimates percentage probabilities for Malignant Risk vs. Benign Likelihood.
2. 📐 **Key Tumor Metrics:** Focuses on Mean Radius, Mean Texture, Mean Perimeter, Mean Area, and Mean Smoothness.
3. 📉 **Visual Analytics:** Interactive horizontal probability bar chart and custom CSS layout.

---

## 🚀 Getting Started

### 📋 Prerequisites
Make sure you have Python 3.8+ installed.

### 📦 Installation
Install all required dependencies using pip:

```bash
pip install streamlit pandas numpy joblib plotly scikit-learn
