# gethub code for Mackey-Glass generation
# https://github.com/mila-iqia/summerschool2015/blob/master/rnn_tutorial/synthetic.py

import collections
import numpy as np
import matplotlib.pyplot as plt

class MackeyGlass:
    """
    generates&predicts Mackey-Glass time-series data
    """

    def __init__(self):
        self._w = np.ones((7, 1)) * 0.1  # +0.1
        print(f"init weights: {self._w}  shape: {self._w.shape}")

    def _generate_data(self, sample_len=5000, tau=17, seed=None):
        """
        Generates the Mackey-Glass time-series.
        :param sample_len: number of samples to generate
        :param tau: Mackey-Glass generator equation parameter
        :param seed: to seed the random generator, can be used to generate the same timeseries at each invocation.
        :return: time:
        """
        delta_t = 10
        history_len = tau * delta_t
        # Initial conditions for the history of the system
        timeseries = 1.2
        if seed is not None:
            np.random.seed(seed)
        time = np.arange(0.0, sample_len, 1)
        history = collections.deque(1.2 * np.ones(history_len) + 0.2 * (np.random.rand(history_len) - 0.5))
        # Preallocate the array for the time-series
        inp = np.zeros((sample_len, 1))
        for timestep in range(sample_len):
            for _ in range(delta_t):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - 0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries
        # Squash timeseries through tanh
        inp = np.tanh(inp - 1)
        samples = inp
        # samples.append(inp)
        return time, samples

    def train(self):
        """
        trains LMS model for self.sample_len time-series samples
        """
        time, series = self._generate_data()
        series = series * 10
        # Training
        sample_len = 5000
        y = np.zeros((sample_len, 1))
        e = np.zeros((sample_len, 1))
        lr = 0.02
        for step in range(time.size - 7):
            t = step + 7
            step_series = series[t-7:t]
            y[t] = np.dot(self._w.transpose(), step_series)
            e[t] = series[t] - y[t]
            self._w = np.array([(self._w[i] + (lr*e[t]*step_series[i])) for i in range(7)])
            print(f"step:{step} weights = {self._w}")
        print(f"step:{step} weights = {self._w}")

    def predict(self):
        pass


if __name__ == "__main__":
    mg = MackeyGlass()
    mg.train()
    mg.predict()