import numpy as np
import matplotlib.pyplot as plt

from ConditionalGAN.GANTrainer import GANTrainer


def main():
    gan_trainer = GANTrainer()

    radius1 = np.random.uniform(1.6, 2.0, 1000)
    phi1 = np.random.uniform(0, 1 * np.pi, 1000)
    x1 = radius1 * np.cos(phi1) + 2
    y1 = radius1 * np.sin(phi1) + 2

    radius2 = np.random.uniform(1, 1.5, 1000)
    phi2 = np.random.uniform(0, 2 * np.pi, 1000)
    x2 = radius2 * np.cos(phi2) - 1
    y2 = radius2 * np.sin(phi2) - 1

    radius3 = np.random.uniform(0, 1, 1000)
    phi3 = np.random.uniform(0, 0.5 * np.pi, 1000)
    x3 = radius3 * np.cos(phi3) - 2
    y3 = radius3 * np.sin(phi3) + 2

    radius4 = np.random.uniform(0, 1, 1000)
    phi4 = np.random.uniform(0, 2 * np.pi, 1000)
    x4 = radius4 * np.cos(phi4) + 3
    y4 = radius4 * np.sin(phi4) - 1

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

    test_label = np.zeros([1000, 4])
    test_label[:250, 0] = 1
    test_label[250:500, 1] = 1
    test_label[500:750, 2] = 1
    test_label[750:, 3] = 1

    x = np.linspace(-7, 7, 100)
    xx, yy = np.meshgrid(x, x)

    for epoch_idx in range(1000):
        input_noise = np.random.uniform(-1, 1, [4000, 32])
        loss_generator, loss_discriminator = gan_trainer.train_(data, label, input_noise)
        print('epoch %02d, gen_loss %.3f, dis_loss %.3f' % (epoch_idx, loss_generator.numpy(), loss_discriminator.numpy()))

        test_input_noise = np.random.uniform(-1, 1, [1000, 32])
        test_concat_input_noise = np.concatenate([test_input_noise, test_label], axis=1)
        test_output = gan_trainer.generator(test_concat_input_noise, training=False)

        # tmp = np.zeros([100, 100])
        # for i in range(100):
        #     tmp_data = np.zeros([100, 2])
        #     tmp_data[:, 0] = xx[i]
        #     tmp_data[:, 1] = yy[i]
        #     tmp[i] = gan_trainer.discriminator(tmp_data, training=False)[:, 0]

        fig, ax = plt.subplots(1, 1)
        # ax.pcolor(xx, yy, tmp, cmap='jet', vmin=0., vmax=1., alpha=0.1)
        ax.scatter(x1, y1, color='blue', alpha=0.2)
        ax.scatter(x2, y2, color='green', alpha=0.2)
        ax.scatter(x3, y3, color='yellow', alpha=0.2)
        ax.scatter(x4, y4, color='red', alpha=0.2)
        ax.scatter(test_output[:250, 0], test_output[:250, 1], color='blue', alpha=0.2, marker="*", edgecolors='black')
        ax.scatter(test_output[250:500, 0], test_output[250:500, 1], color='green', alpha=0.2, marker="*", edgecolors='black')
        ax.scatter(test_output[500:750, 0], test_output[500:750, 1], color='yellow', alpha=0.2, marker="*", edgecolors='black')
        ax.scatter(test_output[750:, 0], test_output[750:, 1], color='red', alpha=0.2, marker="*", edgecolors='black')
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
        fig.savefig('./tmp_%d.png' % epoch_idx)
        plt.close(fig)


def main_gif():
    import imageio
    from PIL import Image

    paths = []
    for i in range(1000):
        img = Image.open('./tmp_%d.png' % i)
        paths.append(img)
    imageio.mimsave('./result.gif', paths, fps=40)


if __name__ == "__main__":
    # main()
    main_gif()
