# Stock Market Sentiment Analysis and Prediction

This project uses sentiment analysis of Reddit posts to predict stock market movements. The model scrapes data from Reddit, processes it, analyzes sentiment using TextBlob, and then trains a machine learning model to predict stock price movements (up or down) based on sentiment. The project is implemented using Python, Jupyter notebooks, and several machine learning libraries.

## Setup Requirements


### Dependencies

This project requires the following Python packages:

- `pandas`
- `numpy`
- `yfinance`
- `praw`
- `textblob`
- `sklearn`

You can install all dependencies by using the `requirements.txt` file.

#### To install dependencies:
     pip install -r requirements.txt

Or individually, you can install the required libraries using pip:
  - `pip install pandas numpy yfinance praw textblob sklearn`

### Reddit API Setup:
You need to create a Reddit API client to access Reddit data. You can do this by registering your application on Reddit's developer site and obtaining the `client_id`, `client_secret`, and `user_agent`. These keys should be securely stored in your environment and never hardcoded in the code.

### Stock Market Data:
The project uses the yfinance library to fetch stock market data. Make sure you have an internet connection when running the code.

### How to Run the Code
1. Data Scraping & Preprocessing
The data scraping script fetches Reddit data using the `PRAW` library and stock market data using the `yfinance` API. This step involves:

- Fetching Reddit posts for sentiment analysis.
- Downloading stock market data (e.g., SPY).
- Preprocessing and cleaning the data.

### 2. Sentiment Analysis
In this step, the project performs sentiment analysis using the TextBlob library on Reddit posts. The sentiment score is calculated for each post, which will later be used as a feature for predicting stock market movement.


### 3. Model Training and Evaluation
The model uses the sentiment data from Reddit and stock market movement data to train a Random Forest classifier to predict whether the stock market will go "Up" or "Down" based on sentiment. This step involves:

- Splitting the data into training and testing sets.
- Training a Random Forest classifier.
- Evaluating the model using classification metrics.

## Evaluation Metrics
After training the model, various classification metrics are displayed, including:

- Accuracy: The percentage of correctly predicted stock movements.
- Precision, Recall, and F1-Score: Performance metrics for each class (Up/Down).

## Improvements and Future Work

- **Data Preprocessing**: Improve text cleaning, handle missing data.
- **Feature Engineering**: Add more features like moving averages, technical indicators.
- **Modeling**: Tune hyperparameters, try other models like XGBoost or LSTM.
- **Sentiment Analysis**: Use advanced models like VADER or BERT.
- **Evaluation**: Address class imbalance, use more evaluation metrics.
- **Time Series**: Try LSTM for better temporal analysis.
- **Visualization**: Visualize sentiment trends with stock prices.
- **Real-Time**: Build a real-time prediction system.

## Explore Complete Project

[https://sandeep5647.github.io/Stock-Movement-Analysis/](https://sandeep5647.github.io/Stock-Movement-Analysis/)