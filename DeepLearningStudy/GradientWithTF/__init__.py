from GradientWithTF.gradient_descent import gradient_descent
from GradientWithTF.gradient_test import gradient_test
import os


def main_gif(dir):
    import imageio
    from PIL import Image

    paths = []
    for i in range(200):
        img = Image.open(os.path.join(dir, 'result_%d.png' % i))
        paths.append(img)
    imageio.mimsave(os.path.join(dir, 'result.gif'), paths, fps=40)


if __name__ == "__main__":

    # gradient_test()

    DIR = './result1'
    gradient_descent(True, dir=DIR)
    main_gif(dir=DIR)
