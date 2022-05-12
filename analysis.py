from collections import defaultdict
import json
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

with open("illinois_net.json") as infile:
    data = json.load(infile)['readings']

def dBm_distance(freq, power, txpower_dBm=20):
    """Convert dBm measurement to distance (roughly).
    Assumes free space haha!!!

    freq: Transmit frequency, in hertz
    power: dBm power
    """
    c = 3e8
    l = c/freq
    txpower_dBm = 20    # 2GHz IEEE standard: 20dBm

    m = l / (4*math.pi*10**((power - txpower_dBm) / 20))
    return m

data_entries = defaultdict(lambda: [])  # Key: (address, name) pairs. Value: a list of (x, y, signal)
x_list = []
y_list = []
for datapoint in data:
    x = datapoint['location']['x']
    y = datapoint['location']['y']
    x_list.append(x)
    y_list.append(y)
    data_list = datapoint['signals']
    for reading in data_list:
        key = (reading['mac'], reading['ssid'])
        #zval = dBm_distance(reading['frequency'], reading['signal'])  # distance estimate (poor)
        #zval = reading['strength']  # dBm direct
        zval = 10**(reading['strength']/10) / 1   # power in milliwatts
        data_entries[key].append((x, y, zval))

# Known: distance from 2 to 3 is 3.2m
d23 = 3.2

plt.scatter(x_list, y_list)
ax = plt.gca()
for i, xy in enumerate(zip(x_list, y_list)):
    ax.annotate(f'{i}', xy=xy, textcoords='data')
plt.show()

d23_px = np.linalg.norm((x_list[2] - x_list[3], y_list[2] - y_list[3]))
scale_factor = d23 / d23_px
for key, v in data_entries.items():
    data_entries[key] = np.array([(x*scale_factor, y*scale_factor, zval) for x, y, zval in v])

import scipy.interpolate
def make_interp(data):
    if len(data) >= 4:
        return scipy.interpolate.LinearNDInterpolator(data[:, :2], data[:, 2])
    return lambda xs, ys: np.zeros(xs.shape)
models = { k: make_interp(data) for k, data in data_entries.items() }

print("Summary:")
print(len(data_entries), "unique signals.")

def prompt():
    command = input("Enter a command: ").strip()
    return command

key1 = ("01:00:70:54:25:2F", "IBM")
key2 = ("01:00:70:54:25:2F", "ARRIS-72E5-5G")

def plot(key):
    plot_data = np.array(data_entries[key])
    model = models[key]

    xs = np.linspace(np.min(plot_data[:,0]), np.max(plot_data[:,0]))
    ys = np.linspace(np.min(plot_data[:,1]), np.max(plot_data[:,1]))
    X, Y = np.meshgrid(xs, ys)
    _X, _Y = X.reshape(-1), Y.reshape(-1)
    C = model(_X, _Y).reshape(X.shape)
    min_color = np.nanmin(plot_data[:, 2])
    max_color = np.nanmax(plot_data[:, 2])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, C, vmin=min_color, vmax=max_color, cmap=cm.coolwarm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(str(key))

keys = list(data_entries.keys())
idx = 0
while True:
    res = input()
    if res:
        key = keys[idx]
        plot(key)
        print(data_entries[key])
        if res == 's':
            fname = '-'.join(key) + '.data'
            data_entries[key].dump(fname)
        idx += 1
        plt.show(block=False)
    else:
        break

while True:
    res = prompt()
    if res == 'x':
        break
    if res == 'ls':
        for i, (addr, name) in enumerate(data_entries.keys()):
            print(f"{i}\t{addr}\t{name}")
