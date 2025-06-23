# 🎮 Pokémon Go User Data Insights

This project analyzes user behavior in **Pokémon Go** using customer data from Summer 2022. Through data-driven techniques such as RFM analysis, lifecycle segmentation, and churn prediction, it offers strategic insights for improving user retention, engagement, and monetization.

📘 Full report: [`Pokemon Go - Data Driven Insights Report.pdf`](./Pokemon%20Go%20-%20Data%20Driven%20Insights%20Report.pdf)  
💻 Python notebook: [`PokemonGo_User Data Insights.py`](./PokemonGo_User%20Data%20Insights.py)

---

## 📌 Objectives

- Identify key player types and behavior patterns  
- Analyze spending and engagement metrics  
- Predict user churn and understand retention drivers  
- Recommend CRM and marketing strategies for Niantic  

---

## 📊 Key Techniques

- **RFM Modeling** (Recency, Frequency, Monetary): customized for both spenders and non-spenders  
- **Customer Segmentation**: Walkers, Catchers, Social Raiders, Miscellaneous  
- **Lifecycle Grids**: engagement vs. recency  
- **Customer Lifetime Value (CLV)** estimation by player type  
- **Churn Prediction** using logistic regression  
- **Strategic Recommendations** based on data insights  

---

## 📈 Insights Summary

- **Social Raiders** are the highest spenders but most likely to churn (40%)  
- **Walkers and Catchers** are more loyal with lower churn (22–23%)  
- **Players aged 26–32** and **low-income groups** show higher CLV  
- **60% of users are male**, and **28 years old** is the average age  
- **77% of players** make only **one transaction**, indicating low repeat monetization  
- Receiving a **Fall Bonus** strongly correlates with reduced churn  

---

## 🔍 Churn Model Results

- **Accuracy**: 63.8%  
- **Precision**: 41.5%  
- **Recall**: 48.7%  
- **AUC Score**: 61.7%  

✅ Important churn predictors:
- Player type (Social Raider increases churn)  
- Recency of session  
- Days since joined  
- Number of PokéStops visited  
- Fall bonus receipt  

---

## 💡 Strategic Recommendations

### 🎯 Retention Tactics
- Personalized rewards for Champions  
- Free campaigns for Catchers and Walkers  
- Promote group events to retain Social Raiders  

### 💰 Monetization Tactics
- Introduce low-cost bundles for new users  
- Create loyalty discounts for high-engagement players  
- Encourage social spending with group-based promotions  

---

## 🛠️ How to Run the Code

# Clone the repo
git clone https://github.com/xuefeiwang001/PokemonGo-User_Data_Driven_Insights.git <br>
cd pokemon-go-data-insights

# Install necessary libraries
pip install pandas numpy matplotlib seaborn scikit-learn

# Run in your preferred Python IDE or Jupyter

---

## 🧪 Tech Stack
- Python 3.8+
- Pandas / NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Logistic Regression
- Stepwise Feature Selection

---

## 📁 Project Structure
- PokemonGo_User Data Insights.py
- Pokemon Go - Data Driven Insights Report.pdf
- README.md

---

## 📄 License
This project is shared under the MIT License.

---

## 👩‍💻 Author
Xuefei Wang
📘 MSc in Data Analytics and AI <br>
🔍 Driven to transform raw data into strategic insights that lead to real impact. <br>
📧 [xuefei.wang001@qq.com]
