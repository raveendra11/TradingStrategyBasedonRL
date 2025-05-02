
def format_price(n):
    return "-$" + format(abs(n), ".2f") if n < 0 else "$" + format(n, ".2f")


def get_state(data, t, window_size):

    d = t - window_size + 1
    block = data[d:t + 1] if d >= 0 else data[0:t + 1]


    if d < 0:
        pad = [data[0]] * abs(d)
        block = pad + list(block)

    state = []
    for i in range(window_size - 1):
        diff = [block[i + 1][j] - block[i][j] for j in range(len(data[0]))]
        state += diff
    return state
