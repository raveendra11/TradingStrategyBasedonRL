import matplotlib.pyplot as plt
from utils import format_price


def plot_results(env, agent):
    data = env.data
    trades = agent.best_trades if hasattr(agent, 'best_trades') else agent.trades

    buy_points = []
    sell_points = []
    profits = []
    profit_dates = []

    total_profit = 0

    for entry in trades:
        if entry[0] == 'buy':
            buy_points.append((entry[1], entry[2]))
        elif entry[0] == 'sell':
            sell_points.append((entry[1], entry[2]))
            total_profit += entry[3]
            profits.append(total_profit)
            profit_dates.append(env.df['Date'][entry[1]])

    fig, ax = plt.subplots(figsize=(14, 6))


    ax.plot(profit_dates, profits, label='Cumulative Profit', color='blue')

    ax.scatter(env.df['Date'][[p[0] for p in buy_points]],
               [p[1] for p in buy_points], marker='^', color='green', label='Buy', s=80)

    ax.scatter(env.df['Date'][[p[0] for p in sell_points]],
               [p[1] for p in sell_points], marker='v', color='red', label='Sell', s=80)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price / Profit")
    ax.set_title(f"Buy/Sell Signals and Profit Curve (Total Profit = {format_price(total_profit)})")
    ax.legend()
    ax.grid(True)

    return fig

