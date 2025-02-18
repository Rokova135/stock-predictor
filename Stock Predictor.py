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
    QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFrame, QSpinBox, QPushButton, QComboBox
)
from PyQt6.QtCore import Qt
import mplcursors

class StockPredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Predictor")
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("background-color: black;")

        main_layout = QVBoxLayout()

        header = QLabel("Stock Predictor")
        header.setStyleSheet("font-size: 48px; font-weight: bold; color: white;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        content_container = QFrame()
        content_container.setStyleSheet("background-color: #333333; border-radius: 10px;")
        content_layout = QHBoxLayout(content_container)

        self.stock_selector = QComboBox()
        self.stock_selector.addItems(["Tesla", "Apple", "Amazon", "Alphabet"])
        self.stock_selector.setStyleSheet("font-size: 20px; color: white; background-color: #444444;")
        content_layout.addWidget(self.stock_selector)

        self.prediction_days_spinbox = QSpinBox()
        self.prediction_days_spinbox.setRange(1, 365)
        self.prediction_days_spinbox.setValue(30)
        self.prediction_days_spinbox.setStyleSheet("font-size: 20px; color: white; background-color: #444444;")
        content_layout.addWidget(self.prediction_days_spinbox)

        update_button = QPushButton("Update Predictions")
        update_button.setStyleSheet("font-size: 20px; color: white; background-color: #666666;")
        update_button.clicked.connect(self.update_predictions)
        content_layout.addWidget(update_button)

        main_layout.addWidget(content_container)

        self.canvas_container = QFrame()
        self.canvas_layout = QHBoxLayout(self.canvas_container)
        main_layout.addWidget(self.canvas_container)

        self.setLayout(main_layout)

        self.update_predictions()

    def update_predictions(self):
        stock_mapping = {
            "Tesla": ("TSLA", "tesla.keras"),
            "Apple": ("AAPL", "apple.keras"),
            "Amazon": ("AMZN", "amazon.keras"),
            "Alphabet": ("GOOG", "alphabet.keras")
        }
        selected_stock = self.stock_selector.currentText()
        stock_ticker, model_filename = stock_mapping[selected_stock]

        data = self.fetch_stock_data(stock_ticker, period="90d")
        actual_prices = data['Close'].dropna().values
        actual_dates = data.index[-len(actual_prices):]

        # Ensure all actual_dates are tz-naive
        actual_dates = actual_dates.tz_localize(None)

        prediction_days = self.prediction_days_spinbox.value()
        predicted_prices = self.predict_stock_prices(data, prediction_days, model_filename)
        prediction_start_date = actual_dates[-1] + pd.Timedelta(days=1)
        future_dates = pd.date_range(start=prediction_start_date, periods=len(predicted_prices), freq="D")

        connected_predictions = np.insert(predicted_prices, 0, actual_prices[-1])
        connected_dates = np.insert(future_dates, 0, actual_dates[-1])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(actual_dates, actual_prices, label="Actual Prices", color='blue')
        ax.plot(connected_dates, connected_predictions, label="Predicted Prices", color='red', linestyle='dashed')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
        plt.xticks(rotation=45, ha='right', color='black')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}"))
        plt.yticks(color='black')
        ax.tick_params(axis='both', labelsize=10, colors='black')
        ax.set_title(f"{selected_stock} - Last 90 Days", color='white', fontsize=16)
        ax.set_xlabel("Date", color='white', fontsize=12)
        ax.set_ylabel("Price (USD)", color='white', fontsize=12)
        ax.legend(facecolor='white', edgecolor='black', fontsize=10, loc='upper left')
        ax.grid(True, color='gray')
        plt.tight_layout()

        cursor = mplcursors.cursor(ax, hover=True)

        # Add hover functionality to show price and date
        def on_hover(event):
            if event.inaxes == ax:
                # Get the x and y data corresponding to the cursor position
                x_data = event.xdata
                y_data = event.ydata

                if x_data is not None and y_data is not None:
                    # Ensure x_data is tz-naive
                    x_data_naive = pd.to_datetime(x_data).tz_localize(None)

                    # Find the closest date and price
                    closest_date_index = np.argmin(np.abs(actual_dates - x_data_naive))
                    closest_date = actual_dates[closest_date_index]
                    closest_price = actual_prices[closest_date_index]


        # Connect the hover event to the function
        fig.canvas.mpl_connect("motion_notify_event", on_hover)

        # Rebuild the canvas after drawing the figure
        for i in reversed(range(self.canvas_layout.count())):
            widget = self.canvas_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        canvas = FigureCanvas(fig)
        self.canvas_layout.addWidget(canvas)

    def fetch_stock_data(self, ticker, period="30d"):
        stock = yf.Ticker(ticker)
        return stock.history(period=period)[['Open', 'High', 'Low', 'Close']]

    def predict_stock_prices(self, data, prediction_days, model_filename):
        try:
            model = tf.keras.models.load_model(model_filename)
        except Exception as e:
            print("Error loading model:", e)
            return np.zeros(prediction_days)

        scaler = MinMaxScaler(feature_range=(0, 1))
        prices = data[['Open', 'High', 'Low', 'Close']].values
        scaled_prices = scaler.fit_transform(prices)
        sequence_length = 30
        test_data = [scaled_prices[i - sequence_length:i] for i in range(sequence_length, len(scaled_prices))]
        test_data = np.array(test_data).reshape(-1, sequence_length, 4)
        predictions = model.predict(test_data[-prediction_days:])
        predictions = scaler.inverse_transform(np.hstack((np.zeros((predictions.shape[0], 3)), predictions)))[:, -1]
        return predictions.flatten()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockPredictorApp()
    window.show()
    sys.exit(app.exec())
