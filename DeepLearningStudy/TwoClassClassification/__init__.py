import numpy as np
import matplotlib.pyplot as plt

from TwoClassClassification.MLP import Network

network = Network('name')

radius1 = np.random.uniform(1, 2, 1000)
phi1 = np.random.uniform(0, 2 * np.pi, 1000)
x1 = radius1 * np.cos(phi1)
y1 = radius1 * np.sin(phi1)

radius2 = np.random.uniform(3, 4, 1000)
phi2 = np.random.uniform(0, 2 * np.pi, 1000)
x2 = radius2 * np.cos(phi2)
y2 = radius2 * np.sin(phi2)

data = np.zeros([2000, 2])
data[:1000, 0] = x1
data[:1000, 1] = y1
data[1000:, 0] = x2
data[1000:, 1] = y2

label = np.zeros([2000, 2])
label[:1000, 0] = 1
label[:1000, 1] = 0
label[1000:, 0] = 0
label[1000:, 1] = 1

x = np.linspace(-5, 5, 100)
xx, yy = np.meshgrid(x, x)

for epoch_idx in range(100):
    loss = network.train_(data, label)
    print('epoch %02d, loss %.3f' % (epoch_idx, loss))

    fig, ax = plt.subplots(1, 1)
    tmp = np.zeros([100, 100])
    for i in range(100):
        tmp_data = np.zeros([100, 2])
        tmp_data[:, 0] = xx[i]
        tmp_data[:, 1] = yy[i]
        tmp[i] = network(tmp_data)[:, 0]
    ax.pcolor(xx, yy, tmp)
    ax.scatter(x1, y1, color='blue')
    ax.scatter(x2, y2, color='red')
    fig.savefig('./tmp_%d.png' % epoch_idx)
    plt.close(fig)
    print()

import imageio
from PIL import Image

paths = []
for i in range(55):
    img = Image.open('./tmp_%d.png' % i)
    paths.append(img)
imageio.mimsave('./result.gif', paths, fps=30)
