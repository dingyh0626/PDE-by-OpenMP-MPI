import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def read_data(path):
    f = open(path, 'rb')
    b = f.read()
    f.close()
    d_len = len(b)
    d_len2 = int(d_len / 8)
    N = int(np.sqrt(d_len2))
    h = 1. / (N - 1)
    data = list(struct.unpack(d_len2 * 'd', b))
    return data, N, h


if __name__ == '__main__':
    data, N, h = read_data('u.data')
    X = []
    Y = []
    Z = []
    for index, z in enumerate(data):
        i = index % N
        j = int(index / N)
        x = i * h
        y = j * h
        X.append(x)
        Y.append(y)
        Z.append(z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, marker='.', alpha=0.2)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

