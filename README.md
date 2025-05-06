# Stock Trading Strategy Using Reinforcement Learning

## Overview

This project implements a stock trading strategy based on Reinforcement Learning using Deep Q Networks (DQN). It allows users to upload market data, train a trading agent, visualize trade decisions, and analyze profits through an interactive Streamlit dashboard.

## Prerequisites

* Python 3.8 or higher
* Virtual Environment (recommended)

## Required Libraries

* numpy
* pandas
* matplotlib
* tensorflow
* streamlit

## Installation

Follow these steps to set up the environment on **Windows**:

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   venv\Scripts\activate.bat
   ```

3. **Install the required packages**:

   ```bash
   pip install numpy pandas matplotlib tensorflow streamlit
   ```

## Project Structure

* `src/agent.py`: Defines the DQN trading agent
* `src/app.py`: Streamlit UI for uploading data and training the agent
* `src/data_loader.py`: Loads and preprocesses stock data
* `src/environment.py`: Simulates the trading environment
* `src/plot.py`: Visualizes trades and profits
* `src/utils.py`: Provides helper functions for state generation and formatting

## How to Run the Project

1. Run the Streamlit application:

   ```bash
   streamlit run .\src\app.py
   ```

2. Open the displayed URL in your browser.

3. Upload a CSV file with stock data (must include `Date`, `Open`, `High`, `Low`, `Close`, `Volume`).

4. Adjust hyperparameters (window size, training episodes, batch size) using the UI sliders.

5. Click **"Train Agent"** to begin training.

6. Once completed, view profit graphs and trade decisions in the dashboard.

## CSV Data Format

Ensure your CSV file is structured like:

```
Date,Open,High,Low,Close,Volume
2023-01-01,100,105,99,104,10000
2023-01-02,104,108,102,107,12000
...
```

## Notes

* Trained models are saved automatically in the `models/` directory.
* The agent visualizes only the **best-performing episode** for clearer analysis.
* For production use, consider adding transaction costs, risk metrics, and advanced RL algorithms.
