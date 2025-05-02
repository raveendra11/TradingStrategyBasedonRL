import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from utils import get_state, format_price


class TradingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.inventory = []
        self.trades = []
        self.best_trades = []

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state):
        state = np.array(state).reshape(1, -1)
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            target_vec = self.model.predict(state.reshape(1, -1), verbose=0)
            target_vec[0][action] = target
            self.model.fit(state.reshape(1, -1), target_vec, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, episodes=5, batch_size=8):
        best_profit = float('-inf')


        for e in range(episodes):
            self.trades = []
            print(f"\n--- Episode {e + 1} ---")
            state = get_state(env.data, 0, env.window_size)
            total_profit = 0
            self.inventory = []

            for t in range(env.window_size, len(env.data)):
                action = self.act(state)
                next_state = get_state(env.data, t, env.window_size)
                reward = 0

                if action == 1:
                    self.inventory.append(env.data[t][3])
                    print(f"Buy at {format_price(env.data[t][3])}")
                    self.trades.append(("buy", t, env.data[t][3]))

                elif action == 2 and len(self.inventory) > 0:
                    bought_price = self.inventory.pop(0)
                    profit = env.data[t][3] - bought_price
                    reward = max(profit, 0)
                    total_profit += profit
                    print(f"Sell at {format_price(env.data[t][3])} | Profit: {format_price(profit)}")
                    self.trades.append(("sell", t, env.data[t][3], profit))

                done = (t == len(env.data) - 1)
                self.remember(state, action, reward, next_state, done)
                state = next_state

                if len(self.memory) > batch_size:
                    self.replay(batch_size)

            print(f"Episode {e + 1} Profit: {format_price(total_profit)}")

            if total_profit > best_profit:
                best_profit = total_profit
                self.best_trades = list(self.trades)
                self.save_model("models/best_trading_model.keras")
                print(f" Best model saved with profit: {format_price(best_profit)}")

    def save_model(self, path="trained_model.keras"):
        self.model.save(path)
        print(f"Model saved to {path}")
