class TradingEnvironment:
    def __init__(self, data, df, window_size):

        self.data = data
        self.df = df
        self.window_size = window_size
        self.state_size = (window_size - 1) * data.shape[1]
        self.action_size = 3
