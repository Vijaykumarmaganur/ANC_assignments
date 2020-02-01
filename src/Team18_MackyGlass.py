# gethub code for Mackey-Glass generation
# https://github.com/mila-iqia/summerschool2015/blob/master/rnn_tutorial/synthetic.py

import collections
import numpy as np
import matplotlib.pyplot as plt

def mackey_glass(sample_len=5000, tau=17, seed=None, n_samples = 1):
    '''
    mackey_glass(sample_len=1000, tau=17, seed = None, n_samples = 1) -> input
    Generate the Mackey Glass time-series. Parameters are:
        - sample_len: length of the time-series in timesteps. Default is 1000.
        - tau: delay of the MG - system. Commonly used values are tau=17 (mild
          chaos) and tau=30 (moderate chaos). Default is 17.
        - seed: to seed the random generator, can be used to generate the same
          timeseries at each invocation.
        - n_samples : number of samples to generate
    '''
    delta_t = 10
    history_len = tau * delta_t
    # Initial conditions for the history of the system
    timeseries = 1.2

    if seed is not None:
        np.random.seed(seed)

    time = np.arange(0.0,sample_len,1)
    # print(time)
    samples = []

    for _ in range(n_samples):
        history = collections.deque(1.2 * np.ones(history_len) + 0.2 * \
                                    (np.random.rand(history_len) - 0.5))
        # Preallocate the array for the time-series
        inp = np.zeros((sample_len,1))

        for timestep in range(sample_len):
            for _ in range(delta_t):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
                             0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries

        # Squash timeseries through tanh
        inp = np.tanh(inp - 1)
        samples = inp
        # samples.append(inp)
    return time, samples

time, series = mackey_glass()
series = series*10
# Training
sample_len = 5000
y = np.zeros((sample_len,1))
e = np.zeros((sample_len,1))
# x[0:6] = series[0:6]
w = np.ones((7,1))*0.1 #+0.1
lr = 0.02
for step in range(time.size-7):
    t = step + 7
    y[t] = w[0]*series[t-1] + w[1]*series[t-2] + w[2]*series[t-3] + w[3]*series[t-4] + w[4]*series[t-5] + w[5]*series[t-6] + w[6]*series[t-7]
    e[t] = series[t] - y[t]
    w[0] = w[0] + (lr*e[t]*series[t-1])
    w[1] = w[1] + (lr*e[t]*series[t-2])
    w[2] = w[2] + (lr*e[t]*series[t-3])
    w[3] = w[3] + (lr*e[t]*series[t-4])
    w[4] = w[4] + (lr*e[t]*series[t-5])
    w[5] = w[5] + (lr*e[t]*series[t-6])
    w[6] = w[6] + (lr*e[t]*series[t-7])
    # print('predicted: ', y[t-1])
    # print('weights: ', w[0:7])
    # print('error: ', e[t])

plt.figure()
plt.subplot(311)
plt.plot(time, series)
plt.ylabel('input')
plt.xlabel('time')
plt.subplot(312)
plt.plot(time,y)
plt.ylabel('predicted')
plt.xlabel('time')
plt.subplot(313)
plt.plot(time,e)
plt.ylabel('error')
plt.xlabel('time')
plt.show()
