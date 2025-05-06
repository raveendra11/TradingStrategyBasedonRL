# Stock Trading Strategy Using Reinforcement Learning

## Overview

This project implements a stock trading strategy based on Reinforcement Learning using Deep Q Networks (DQN). It allows users to upload market data, train a trading agent, visualize trade decisions, and analyze profits through an interactive Streamlit dashboard.

## Prerequisites

* Python 3.8 or higher
* Virtual Environment (optional but recommended)

## Required Libraries

* numpy
* pandas
* matplotlib
* tensorflow
* streamlit

## Installation

Follow these steps to set up the environment:

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: .\venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install numpy pandas matplotlib tensorflow streamlit
   ```

## Project Structure

* `agent.py`: Defines the DQN trading agent.
* `app.py`: Streamlit UI for uploading data and training the agent.
* `data_loader.py`: Utility to load and preprocess stock data.
* `environment.py`: Sets up the trading environment.
* `plot.py`: Generates visualizations of trades and profits.
* `utils.py`: Helper functions for state calculations and price formatting.

## How to Run the Project

1. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. Open the displayed URL in your browser.

3. Upload a CSV file with stock data containing columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

4. Adjust hyperparameters (window size, training episodes, batch size) through the provided sliders.

5. Click `Train Agent` to start training.

6. Once training completes, view the results and profit visualizations directly in the dashboard.

## CSV Data Format

Ensure your CSV file matches this format:

```
Date,Open,High,Low,Close,Volume
2023-01-01,100,105,99,104,10000
...
```

## Notes

* Trained models are automatically saved in the `models/` directory.
* The application visualizes the best-performing episode to analyze trading decisions.

