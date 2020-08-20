import numpy as np
import matplotlib.pyplot as plt

from VanillaGAN.GANTrainer import GANTrainer


def main():
    gan_trainer = GANTrainer()

    radius1 = np.random.uniform(1.6, 2.0, 1000)
    phi1 = np.random.uniform(0, 1 * np.pi, 1000)
    x1 = radius1 * np.cos(phi1) + 2
    y1 = radius1 * np.sin(phi1) + 2

    data = np.zeros([1000, 2])
    data[:1000, 0] = x1
    data[:1000, 1] = y1

    x = np.linspace(-7, 7, 100)
    xx, yy = np.meshgrid(x, x)

    for epoch_idx in range(1000):
        input_noise = np.random.uniform(-1, 1, [1000, 32])
        loss_generator, loss_discriminator = gan_trainer.train_(data, input_noise)
        print('epoch %02d, gen_loss %.3f, dis_loss %.3f' % (epoch_idx, loss_generator.numpy(), loss_discriminator.numpy()))

        test_input_noise = np.random.uniform(-1, 1, [1000, 32])
        test_output = gan_trainer.generator(test_input_noise, training=False)

        tmp = np.zeros([100, 100])
        for i in range(100):
            tmp_data = np.zeros([100, 2])
            tmp_data[:, 0] = xx[i]
            tmp_data[:, 1] = yy[i]
            tmp[i] = gan_trainer.discriminator(tmp_data, training=False)[:, 0]

        fig, ax = plt.subplots(1, 1)
        ax.pcolor(xx, yy, tmp, cmap='jet', vmin=0., vmax=1., alpha=0.1)
        ax.scatter(x1, y1, color='blue', alpha=0.2)
        ax.scatter(test_output[:, 0], test_output[:, 1], color='red', alpha=0.2)
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
    main()
    main_gif()
