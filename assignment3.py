__author__ = 'geoffrey'


import numpy
import scipy.io
import pylab
from matplotlib.legend_handler import HandlerLine2D
from sklearn.decomposition import PCA
from sklearn.utils import shuffle


if __name__ == '__main__':

    data = scipy.io.loadmat("hw4pca.dat")
    data_array = data['xp']

    # shuffle the data
    shuffled_array = shuffle(data_array, random_state=125)

    # determine the training and testing splits
    training_set_size = shuffled_array.shape[0] * 0.8
    data_set_size = shuffled_array.shape[0]

    shuffled_array = numpy.split(shuffled_array, [training_set_size, data_set_size])

    training_array = shuffled_array[0]
    testing_array = shuffled_array[1]

    dimensions = pylab.arange(249, 0, -1)
    train_reconstruction_error = numpy.zeros((249,), dtype=numpy.float)
    test_reconstruction_error = numpy.zeros((249,), dtype=numpy.float)

    for d in dimensions:

        pca = PCA(d)
        pca.fit(training_array)

        # training reconstruction error
        scores = pca.transform(training_array)
        reconstruct = pca.inverse_transform(scores)

        residual = numpy.sum((training_array - reconstruct) ** 2) / training_set_size
        train_reconstruction_error[249 - d] = residual

        # testing reconstruction error
        scores = pca.transform(testing_array)
        reconstruct = pca.inverse_transform(scores)

        residual = numpy.sum((testing_array - reconstruct) ** 2) / (data_set_size - training_set_size)
        test_reconstruction_error[249 - d] = residual


    pylab.plot(dimensions, train_reconstruction_error, label="training error")
    pylab.plot(dimensions, test_reconstruction_error, label="testing error")
    pylab.legend()

    pylab.ylabel('reconstruction mse')
    pylab.xlabel('dimensions')
    pylab.show()

    numpy.savetxt('data.txt', data_array)