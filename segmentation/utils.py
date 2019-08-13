
import matplotlib.pyplot as plt


def save_tensor_png(t, path):
    cmap = plt.cm.jet
    plt.imsave(path, t, cmap=cmap)
