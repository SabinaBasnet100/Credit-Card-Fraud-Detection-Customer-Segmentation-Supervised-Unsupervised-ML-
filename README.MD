# 💳 Credit Card Fraud Detection & Customer Segmentation (R Project)

This project explores multiple business problems using machine learning and data mining techniques applied to a credit card dataset. It combines **supervised learning**, **unsupervised clustering**, and a **hybrid anomaly detection method** to deliver fraud prevention and customer segmentation insights.

---

## 📊 Dataset Overview

- **Rows:** 6,146  
- **Columns:** 13  
- **Source:** Bank transaction records  
- **Features include:**  
  - Card type and brand  
  - Presence of chip  
  - Credit limit  
  - Number of cards issued  
  - Card presence on dark web  
  - Account open date  

---

## 🎯 Business Problems Addressed

### 1. Fraud Detection (Supervised - Logistic Regression)
- **Objective:** Predict whether a card appears on the dark web.
- **Target Variable:** `card_on_dark_web` (Yes/No)
- **Model:** Logistic Regression
- **Key Findings:**  
  - Achieved **85% accuracy**  
  - `has_chip` reduces fraud likelihood  
  - Higher credit limits = higher risk

---

### 2. Customer Segmentation (Unsupervised - K-Means Clustering)
- **Objective:** Group customers based on usage and credit profiles
- **Features Used:** `credit_limit`, `num_cards_issued`
- **Clusters Identified:**  
  - High-value customers  
  - Mid-tier users  
  - Low-value minimal users
- **Impact:** Enables personalized marketing and better retention strategies

---

### 3. Enhanced Fraud Detection (Mixed - Random Forest + Isolation Forest)
- **Objective:** Combine supervised classification with anomaly detection for broader fraud detection
- **Techniques Used:**  
  - Random Forest Classifier  
  - Isolation Forest  
- **Impact:**  
  - Improved fraud detection accuracy to **90%**  
  - Isolation Forest detected **5% more** high-risk anomalies  
  - Balanced data using ROSE to handle class imbalance

---

## 📁 Project Structure

card-fraud-detection-ml/ ├── credit_card_fraud_detection.R # Full R code for modeling and analysis ├── Assignment 03.Sabina.ML.docx # Documentation of methods and conclusions └── README.md # This file

---

## 🧠 Key Techniques

- Logistic Regression
- Random Forest
- Isolation Forest
- K-Means Clustering
- Silhouette Score
- ROC Curve & AUC
- Balanced Sampling using ROSE
- Cross-validation with caret

---

## 📌 Conclusions & Insights

- **Supervised learning** is effective and interpretable for fraud prediction.
- **K-Means** helped discover hidden customer groups for targeting.
- **Mixed models** improved fraud detection by combining classification and anomaly detection.
- **Balanced sampling** is critical in fraud analysis due to class imbalance.

---

## 👩‍💻 Author

**Sabina Basnet**   
🔗 [LinkedIn Profile](https://www.linkedin.com/in/sabinabasnet)

