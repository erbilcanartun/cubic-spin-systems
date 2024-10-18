import numpy
import mpmath


def interaction(t):
        return mpmath.log(t[0, 0] / t[0, 1]) / 2

def interaction_absolute_average(matrices):
    interactions = []
    for i in range(len(matrices)):
        interactions.append(interaction(matrices[i]))
    return numpy.mean(numpy.absolute(interactions))

def normalized_tracked_interaction(matrices, index):
    interactions = []
    for i in range(len(matrices)):

        interactions.append(interaction(matrices[i]))
    javr = numpy.mean(numpy.absolute(interactions))

    return numpy.float64(interactions[index]) / float(javr)
