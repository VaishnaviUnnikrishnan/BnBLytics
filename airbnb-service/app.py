from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import
from matplotlib import pyplot as plt

import io
import base64
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, accuracy_score

# Initialize app
app = Flask(__name__)
app.config['DEBUG'] = False
# Load model and encoder
with open('model/price_prediction_model.pkl', 'rb') as f:
    model, encoder = pickle.load(f)

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Predict route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predicted_price = None
    graph1 = None
    graph2 = None
    graph3 = None
    graph4 = None
    metrics = {}

    room_types = ['Private room', 'Entire home/apt', 'Shared room']

    if request.method == 'POST':
        room_type = request.form['room_type']
        minimum_nights = int(request.form['minimum_nights'])

        # Encode room type
        room_type_encoded = encoder.transform([room_type])[0]

        # Create input array
        X_input = np.array([room_type_encoded, minimum_nights]).reshape(1, -1)

        # Predict
        predicted_price = model.predict(X_input)[0]

        # Dummy true value (for demo)
        y_true = np.array([predicted_price * 0.95])
        y_pred = np.array([predicted_price])

        # Metrics Calculation
        mae = mean_absolute_error(y_true, y_pred)
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        threshold = 100
        y_true_class = (y_true > threshold).astype(int)
        y_pred_class = (y_pred > threshold).astype(int)

        precision = precision_score(y_true_class, y_pred_class, zero_division=1)
        recall = recall_score(y_true_class, y_pred_class, zero_division=1)
        f1 = f1_score(y_true_class, y_pred_class, zero_division=1)
        accuracy = accuracy_score(y_true_class, y_pred_class)

        metrics = {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R² Score': round(r2, 2),
            'Precision': round(precision, 2),
            'Recall': round(recall, 2),
            'F1 Score': round(f1, 2),
            'Accuracy': round(accuracy, 2)
        }

        # ------------------- Graphs -------------------

        # Graph 1: Bar Chart (True vs Predicted)
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        fig1.patch.set_facecolor('#1c1c1c')
        ax1.set_facecolor('#2e2e2e')
        bars = ax1.bar(['True Price', 'Predicted Price'], [y_true[0], y_pred[0]], color=['red', 'black'])
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f"${yval:.2f}", ha='center', color='white')
        ax1.set_title('True vs Predicted Price', color='white')
        ax1.set_ylabel('Price ($)', color='white')
        ax1.tick_params(colors='white')
        ax1.grid(axis='y', color='gray')
        ax1.spines['bottom'].set_color('white')
        ax1.spines['left'].set_color('white')

        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', bbox_inches='tight', facecolor=fig1.get_facecolor())
        buf1.seek(0)
        graph1 = base64.b64encode(buf1.getvalue()).decode('utf-8')
        plt.close(fig1)

        # Graph 2: Scatter Plot
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        fig2.patch.set_facecolor('#1c1c1c')
        ax2.set_facecolor('#2e2e2e')
        ax2.scatter([1], y_true, color='black', label='True Price', s=100)
        ax2.scatter([1], y_pred, color='red', label='Predicted Price', s=100)
        ax2.set_title('Prediction Scatter Plot', color='white')
        ax2.set_xlabel('X Axis Label', color='white')
        ax2.set_ylabel('Y Axis Label', color='white')
        ax2.legend(facecolor='#2e2e2e', edgecolor='white', fontsize=10, loc='best', labelcolor='white')
        ax2.grid(True, color='gray')
        ax2.tick_params(colors='white')
        ax2.spines['bottom'].set_color('white')
        ax2.spines['left'].set_color('white')

        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', bbox_inches='tight', facecolor=fig2.get_facecolor())
        buf2.seek(0)
        graph2 = base64.b64encode(buf2.getvalue()).decode('utf-8')
        plt.close(fig2)

        # Graph 3: Metrics Bar Plot
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        fig3.patch.set_facecolor('#1c1c1c')
        ax3.set_facecolor('#2e2e2e')
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        bars = ax3.bar(metric_names, metric_values, color='#FF0000')
        ax3.set_title('Model Metrics', color='white', fontsize=16)
        ax3.tick_params(colors='white')
        ax3.set_ylim(0, max(metric_values) + 1)
        ax3.spines['bottom'].set_color('white')
        ax3.spines['left'].set_color('white')
        ax3.grid(axis='y', color='gray')
        plt.xticks(rotation=45)

        for bar in bars:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}", ha='center', color='white', fontsize=10)

        buf3 = io.BytesIO()
        fig3.savefig(buf3, format='png', bbox_inches='tight', facecolor=fig3.get_facecolor())
        buf3.seek(0)
        graph3 = base64.b64encode(buf3.getvalue()).decode('utf-8')
        plt.close(fig3)

        # Graph 4: RMSE vs MAE (Comparison)
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        fig4.patch.set_facecolor('#1c1c1c')
        ax4.set_facecolor('#2e2e2e')
        ax4.bar(['RMSE', 'MAE'], [rmse, mae], color=['red', 'black'])
        ax4.set_title('RMSE vs MAE', color='white')
        ax4.grid(axis='y', color='gray')
        ax4.tick_params(colors='white')
        ax4.spines['bottom'].set_color('white')
        ax4.spines['left'].set_color('white')

        buf4 = io.BytesIO()
        fig4.savefig(buf4, format='png', bbox_inches='tight', facecolor=fig4.get_facecolor())
        buf4.seek(0)
        graph4 = base64.b64encode(buf4.getvalue()).decode('utf-8')
        plt.close(fig4)

    return render_template('predict.html',
                           predicted_price=predicted_price,
                           graph1=graph1,
                           graph2=graph2,
                           graph3=graph3,
                           graph4=graph4,
                           metrics=metrics,
                           room_types=room_types)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
