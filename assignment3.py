__author__ = 'geoffrey'


import numpy
import scipy.io
import pylab
from sklearn.decomposition import PCA


if __name__ == '__main__':

    data = scipy.io.loadmat("hw4pca.dat")
    data_array = data['xp']

    dimensions = pylab.arange(249, 1, -1)
    reconstruction_error = numpy.zeros((248,),dtype=numpy.float)

    for d in dimensions:

        pca = PCA(d)
        pca.fit(data_array)
        scores = pca.transform(data_array)
        reconstruct = pca.inverse_transform(scores)

        residual = numpy.sum((data_array - reconstruct) ** 2) / data_array.shape[0]
        reconstruction_error[249 - d] = residual

    pylab.plot(dimensions, reconstruction_error)
    pylab.ylabel('reconstruction mse')
    pylab.xlabel('dimensions')
    pylab.show()

    numpy.savetxt('data.txt', data_array)