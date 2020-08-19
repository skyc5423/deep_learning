import numpy as np
import matplotlib.pyplot as plt

from MultiClassClassification.MLP import Network

network = Network('name')

radius1 = np.random.uniform(1, 2, 1000)
phi1 = np.random.uniform(0, 2 * np.pi, 1000)
x1 = radius1 * np.cos(phi1)
y1 = radius1 * np.sin(phi1)

radius2 = np.random.uniform(3, 4, 1000)
phi2 = np.random.uniform(0, 2 * np.pi, 1000)
x2 = radius2 * np.cos(phi2)
y2 = radius2 * np.sin(phi2)

radius3 = np.random.uniform(0, 1, 1000)
phi3 = np.random.uniform(0, 2 * np.pi, 1000)
x3 = radius3 * np.cos(phi3) + 2
y3 = radius3 * np.sin(phi3) + 6

radius4 = np.random.uniform(0, 2, 1000)
phi4 = np.random.uniform(0, 2 * np.pi, 1000)
x4 = radius4 * np.cos(phi4) + 6
y4 = radius4 * np.sin(phi4) + 3

data = np.zeros([4000, 2])
data[:1000, 0] = x1
data[:1000, 1] = y1
data[1000:2000, 0] = x2
data[1000:2000, 1] = y2
data[2000:3000, 0] = x3
data[2000:3000, 1] = y3
data[3000:4000, 0] = x4
data[3000:4000, 1] = y4

label = np.zeros([4000, 4])
label[:1000, 0] = 1
label[1000:2000, 1] = 1
label[2000:3000, 2] = 1
label[3000:4000, 3] = 1

x = np.linspace(-5, 9, 100)
xx, yy = np.meshgrid(x, x)

# for epoch_idx in range(300):
#     loss = network.train_(data, label)
#     print('epoch %02d, loss %.3f' % (epoch_idx, loss))
#
#     fig, ax = plt.subplots(1, 1)
#     tmp = np.zeros([100, 100, 4])
#     for i in range(100):
#         tmp_data = np.zeros([100, 2])
#         tmp_data[:, 0] = xx[i]
#         tmp_data[:, 1] = yy[i]
#         tmp[i] = network(tmp_data)[:]
#     tmp = np.argmax(tmp, axis=2)
#     ax.pcolor(xx, yy, tmp, cmap='jet', alpha=0.5)
#     ax.scatter(x1, y1, color='blue', alpha=0.1)
#     ax.scatter(x2, y2, color='green', alpha=0.1)
#     ax.scatter(x3, y3, color='yellow', alpha=0.1)
#     ax.scatter(x4, y4, color='red', alpha=0.1)
#     fig.savefig('./tmp_%d.png' % epoch_idx)
#     plt.close(fig)
#     print()

import imageio
from PIL import Image

paths = []
for i in range(100):
    img = Image.open('./tmp_%d.png' % i)
    paths.append(img)
imageio.mimsave('./result.gif', paths, fps=30)
