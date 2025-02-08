import sys
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFrame,
    QSpinBox,
    QPushButton,
)
from PyQt6.QtCore import Qt


class StockPredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Predictor")
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("background-color: black;")

        # Main layout
        main_layout = QVBoxLayout()

        # Header
        header = QLabel("Stock Predictor")
        header.setStyleSheet("font-size: 48px; font-weight: bold; color: white;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        # Content layout
        content_container = QFrame()
        content_container.setStyleSheet("background-color: #333333; border-radius: 10px;")
        content_layout = QHBoxLayout(content_container)

        # Stock name label
        stock_name = QLabel("Tesla")
        stock_name.setStyleSheet("font-size: 36px; font-weight: bold; color: white;")
        stock_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(stock_name)

        # Spinbox to choose number of prediction days
        self.prediction_days_spinbox = QSpinBox()
        self.prediction_days_spinbox.setRange(1, 365)
        self.prediction_days_spinbox.setValue(30)
        self.prediction_days_spinbox.setStyleSheet("font-size: 20px; color: white; background-color: #444444;")
        content_layout.addWidget(self.prediction_days_spinbox)

        # Button to update predictions
        update_button = QPushButton("Update Predictions")
        update_button.setStyleSheet("font-size: 20px; color: white; background-color: #666666;")
        update_button.clicked.connect(self.update_predictions)
        content_layout.addWidget(update_button)

        # Add content layout to the main layout
        main_layout.addWidget(content_container)

        # Placeholder for the Matplotlib plot canvas
        self.canvas_container = QFrame()
        self.canvas_layout = QHBoxLayout(self.canvas_container)
        main_layout.addWidget(self.canvas_container)

        self.setLayout(main_layout)

        # Initially display predictions for the default number of days
        self.update_predictions()

    def update_predictions(self):
        # Fetch stock data
        stock_ticker = "TSLA"
        data = self.fetch_stock_data(stock_ticker, period="90d")
        actual_prices = data['Close'].dropna().values
        actual_dates = data.index[-len(actual_prices):]

        # Get user-selected prediction days
        prediction_days = self.prediction_days_spinbox.value()

        # Generate model predictions
        predicted_prices = self.predict_stock_prices(actual_prices, prediction_days)
        prediction_start_date = actual_dates[-1] + pd.Timedelta(days=1)
        future_dates = pd.date_range(start=prediction_start_date, periods=len(predicted_prices), freq="D")

        # Ensure continuity by appending last actual price to predictions
        connected_predictions = np.insert(predicted_prices, 0, actual_prices[-1])
        connected_dates = np.insert(future_dates, 0, actual_dates[-1])

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot actual prices
        ax.plot(actual_dates, actual_prices, label="Actual Prices", color='blue')

        # Plot connected predicted prices
        ax.plot(connected_dates, connected_predictions, label="Predicted Prices", color='red', linestyle='dashed')

        # **Fixing X-Axis (Dates)**
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))  # Format like "Feb 06"
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))  # Show every 10 days
        plt.xticks(rotation=45, ha='right', color='black')  # Rotate labels and set black color

        # **Fixing Y-Axis (Prices)**
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}"))  # Format prices as "$123.45"
        plt.yticks(color='black')

        # Adjust tick parameters for better visibility
        ax.tick_params(axis='both', labelsize=10, colors='black')  # Ensure ticks are black

        # **Graph Styling**
        ax.set_title("Tesla - Last 90 Days", color='white', fontsize=16)
        ax.set_xlabel("Date", color='white', fontsize=12)
        ax.set_ylabel("Price (USD)", color='white', fontsize=12)
        ax.legend(facecolor='black', edgecolor='white', fontsize=10, loc='upper left')
        ax.grid(True, color='gray')

        # Automatically adjust layout to prevent cutting off of labels
        plt.tight_layout()

        # Clear previous graph and update
        for i in reversed(range(self.canvas_layout.count())):
            widget = self.canvas_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Add new Matplotlib plot to the layout
        canvas = FigureCanvas(fig)
        self.canvas_layout.addWidget(canvas)

    def fetch_stock_data(self, ticker, period="30d"):
        stock = yf.Ticker(ticker)
        return stock.history(period=period)

    def predict_stock_prices(self, prices, prediction_days):
        try:
            model = tf.keras.models.load_model("model.keras")
        except Exception as e:
            print("Error loading model:", e)
            return np.zeros(prediction_days)

        if len(prices) == 0:
            print("Error: No valid stock prices available for prediction.")
            return np.zeros(1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        prices = np.array(prices).reshape(-1, 1)
        scaled_prices = scaler.fit_transform(prices)

        sequence_length = 30  # The same as training
        test_data = []
        for i in range(sequence_length, len(scaled_prices)):
            test_data.append(scaled_prices[i - sequence_length:i, 0])
        test_data = np.array(test_data).reshape(-1, sequence_length, 1)

        predictions = model.predict(test_data[-prediction_days:])
        return scaler.inverse_transform(predictions).flatten()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockPredictorApp()
    window.show()
    sys.exit(app.exec())
