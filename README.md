# 🏠 BnBLytics - Airbnb Price Prediction and Analysis

**BnBLytics** is a data-driven Airbnb analytics platform designed to visualize Airbnb listing trends in New York City using Tableau, and dynamically predict listing prices using a Flask-based web application powered by an XGBoost regression model.

---

## 🚀 Features

- 🧠 **Airbnb Price Predictor**
  - Select room type and input minimum nights to predict listing price.
  - View true vs predicted price comparisons.
  - Analyze prediction accuracy via scatter and bar plots.

- ⚙️ **ML Model Service**
  - XGBoost model trained on Airbnb NYC dataset.
  - Real-time price prediction with performance metrics and feature importance visualization.

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS (Bootstrap), JavaScript
- **Backend:** Flask (Python)
- **Machine Learning:** XGBoost, scikit-learn, Pandas, Matplotlib
- **Repository:** GitHub

---

## 🧾 Project Structure

```bash
BnBLytics/
├── airbnb-service/           # Flask frontend for user input and predictions
├── modelari-service/         # Trained model and prediction scripts
├── README.md                 # Project documentation
````

---

## 🧪 Getting Started

### Prerequisites

* Python 3.8+
* pip (Python package manager)
* Git

### Installation

1. Clone the repository:

```bash
git clone https://github.com/VaishnaviUnnikrishnan/BnBLytics.git
cd BnBLytics
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
cd airbnb-service
python app.py
```

The app will be available at `http://127.0.0.1:5000`.

---

## 📈 Model Overview

* **Input Features:**

  * Room Type (categorical)
  * Minimum Nights (integer)

* **Target:**

  * Predicted Price (USD)

* **Evaluation Metrics:**

  * R² Score, F1 Score, Recall, Support
  * True vs Predicted Price Chart
  * Prediction Scatter Plot
  * Feature Importance Bar Chart

---

## 📷 Sample Output

![Screenshot 2025-04-28 183227](https://github.com/user-attachments/assets/8d2820eb-81f3-4f03-8d9f-c42fa3a32099)

![Screenshot 2025-04-28 183259](https://github.com/user-attachments/assets/d4a2d134-6236-47e1-acb3-614d8f497a85)

![Screenshot 2025-04-28 183346](https://github.com/user-attachments/assets/6b813188-7da1-4286-ae62-b877b3eeaf54)




---



