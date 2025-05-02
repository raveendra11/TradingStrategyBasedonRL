import streamlit as st
import matplotlib.pyplot as plt
from data_loader import load_stock_data
from agent import TradingAgent
from environment import TradingEnvironment
from plot import plot_results
from tensorflow.keras.models import load_model

st.title("Trading Strategy based on Reinforcement Learning")

csv_file = st.file_uploader("Upload stock CSV file", type=["csv"])

window_size = st.slider("Select window size", min_value=2, max_value=20, value=10)
episodes = st.slider("Training episodes", min_value=1, max_value=100, value=5)
batch_size = st.slider("Batch size", min_value=2, max_value=64, value=8)

if "agent" not in st.session_state:
    st.session_state.agent = None
if "env" not in st.session_state:
    st.session_state.env = None

if csv_file and st.button("Train Agent"):
    df, data = load_stock_data(csv_file)

    st.subheader("Stock Price Trend")
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Close'])
    ax.set_title("Stock Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    st.pyplot(fig)

    env = TradingEnvironment(data, df, window_size)
    agent = TradingAgent(state_size=env.state_size, action_size=env.action_size)

    with st.spinner("Training agent..."):
        agent.train(env, episodes=episodes, batch_size=batch_size)
        agent.save_model("models/best_trading_model.keras")

    st.session_state.agent = agent
    st.session_state.env = env

    st.success("Training completed!")
    st.pyplot(plot_results(env, agent))

