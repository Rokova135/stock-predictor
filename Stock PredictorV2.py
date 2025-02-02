import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import yfinance as yf
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFrame,
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
        header.setStyleSheet("font-size: 128px; font-weight: bold; color: white; background-color: black;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        # Content layout
        content_container = QFrame()
        content_container.setStyleSheet("background-color: #333333; border-radius: 10px;")
        content_layout = QHBoxLayout(content_container)

        # Stock name label
        stock_name = QLabel("S&P 500")
        stock_name.setStyleSheet("font-size: 128px; font-weight: bold; color: white; background-color: black;")
        stock_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(stock_name)

        # Fetch stock data
        sp500 = yf.Ticker("^GSPC")  # ^GSPC is the ticker symbol for the S&P 500
        data = sp500.history(period="1y")  # Fetch 1 year of historical data

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data['Close'], label="S&P 500 Close Price")
        ax.set_title("S&P 500 - Last 1 Year", color='black', fontsize=16)
        ax.set_xlabel("Date", color='black', fontsize=12)
        ax.set_ylabel("Price (USD)", color='black', fontsize=12)
        ax.legend(facecolor='white', edgecolor='black', fontsize=10, loc='upper left')


        # Customize ticks and labels for better visibility
        ax.tick_params(axis='both', colors='black')  # Change tick colors to white
        ax.grid(True, color='white')  # Set grid color to white

        # Add Matplotlib plot to the layout
        canvas = FigureCanvas(fig)
        content_layout.addWidget(canvas)

        # Add content layout to the main layout
        main_layout.addWidget(content_container)

        # Set main layout to the window
        self.setLayout(main_layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockPredictorApp()
    window.show()
    sys.exit(app.exec())
