import numpy as np
import matplotlib.pyplot as plt
from random import sample

def plot_examples(X,N):
    """
    plots a grid of NxN images from rows of X
    :param X: the data. Assumed as [N_samples, 784]
    :param N: width of the grid to plot on
    :return: None
    """
    f, axarr = plt.subplots(N,N)
    ind = iter(list(np.random.choice(X.shape[0], N**2)))
    for i in range(N):
        for j in range(N):
            im = np.reshape(X[next(ind)], [28,28])
            axarr[i,j].imshow(im,cmap='gray')
            plt.setp(axarr[i,j].get_xticklabels(), visible=False)
            plt.setp(axarr[i,j].get_yticklabels(), visible=False)
    f.subplots_adjust(hspace=0)
    f.subplots_adjust(wspace=0)

def shuffle_datasets(N):
    """
    Get N data samples
    :param N:
    :return:
    """
    X1 = np.load('data/full%2Fnumpy_bitmap%2Fcar.npy')
    X1 = X1[np.random.choice(X1.shape[0], N)]

    return (X1>(255./2.)).astype(np.int16)

if __name__ == "__main__":
    X = shuffle_datasets(10000)
    plot_examples(X, 5)