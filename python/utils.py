import numpy
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt


def __rolling_window(data, window_size):
    """
    Rolling window: take window with definite size through the array

    :param data: array-like
    :param window_size: size
    :return: the sequence of windows

    Example: data = array(1, 2, 3, 4, 5, 6), window_size = 4
        Then this function return array(array(1, 2, 3, 4), array(2, 3, 4, 5), array(3, 4, 5, 6))
    """
    shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
    strides = data.strides + (data.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def __cvm(subindices, total_events):
    """
    Compute Cramer-von Mises metric.
    Compared two distributions, where first is subset of second one.
    Assuming that second is ordered by ascending

    :param subindices: indices of events which will be associated with the first distribution
    :param total_events: count of events in the second distribution
    :return: cvm metric
    """
    # here we compute the same expression (using algebraic expressions for them).
    n_subindices = float(len(subindices))
    subindices = numpy.array([0] + sorted(subindices) + [total_events], dtype='int')
    # via sum of the first squares
    summand1 = total_events * (total_events + 1) * (total_events + 0.5) / 3. / (total_events ** 3)
    left_positions = subindices[:-1]
    right_positions = subindices[1:]

    values = numpy.arange(len(subindices) - 1)

    summand2 = values * (right_positions * (right_positions + 1) - left_positions * (left_positions + 1)) / 2
    summand2 = summand2.sum() * 1. / (n_subindices * total_events * total_events)

    summand3 = (right_positions - left_positions) * values ** 2
    summand3 = summand3.sum() * 1. / (n_subindices * n_subindices * total_events)

    return summand1 + summand3 - 2 * summand2


def compute_cvm(predictions, masses, n_neighbours=200, step=50):
    """
    Computing Cramer-von Mises (cvm) metric on background events: take average of cvms calculated for each mass bin.
    In each mass bin global prediction's cdf is compared to prediction's cdf in mass bin.

    :param predictions: array-like, predictions
    :param masses: array-like, in case of Kaggle tau23mu this is reconstructed mass
    :param n_neighbours: count of neighbours for event to define mass bin
    :param step: step through sorted mass-array to define next center of bin
    :return: average cvm value
    """
    predictions = numpy.array(predictions)
    masses = numpy.array(masses)
    assert len(predictions) == len(masses)

    # First, reorder by masses
    predictions = predictions[numpy.argsort(masses)]

    # Second, replace probabilities with order of probability among other events
    predictions = numpy.argsort(numpy.argsort(predictions))

    # Now, each window forms a group, and we can compute contribution of each group to CvM
    cvms = []
    for window in __rolling_window(predictions, window_size=n_neighbours)[::step]:
        cvms.append(__cvm(subindices=window, total_events=len(predictions)))
    return numpy.mean(cvms)


def get_roc_curves(labels, probas, curve_labels, save_path=None, show=True):
    """
    Creates roc curve for each class vs rest.
    :param labels: ndarray, shape = [n_samples, n_classes], labels for the each class.
    1 - if a sample belongs to the class, 0 - otherwise.
    :param probas: ndarray, shape = [n_samples, n_classes], predicted probabilities.
    :param curve_labels: array of strings , shape = [n_classes], labels of the curves.
    :param save_path: string, path to a directory where the figure will saved. If None the figure will not be saved.
    :param show: boolean, if true the figure will be displayed.
    """

    plt.figure(figsize=(10,7))

    for num in range(probas.shape[1]):

        roc_auc = roc_auc_score(labels[:, num], probas[:, num])
        fpr, tpr, _ = roc_curve(labels[:, num], probas[:, num])

        plt.plot(tpr, 1.-fpr, label=curve_labels[num] + ', %.2f' % roc_auc, linewidth=2)

    plt.title("ROC Curves", size=15)
    plt.xlabel("Signal efficiency", size=15)
    plt.ylabel("Background rejection", size=15)
    plt.legend(loc='best',prop={'size':15})
    plt.xticks(numpy.arange(0, 1.01, 0.1), size=15)
    plt.yticks(numpy.arange(0, 1.01, 0.1), size=15)


    if save_path != None:
        plt.savefig(save_path + "/overall_roc_auc.png")

    if show == True:
        plt.show()

    plt.clf()
    plt.close()

def get_roc_auc_matrix(labels, probas, axis_labels, save_path=None, show=True):

    """
    Calculate class vs class roc aucs matrix.
    :param labels: ndarray, shape = [n_samples, n_classes], labels for the each class.
    1 - if a sample belongs to the class, 0 - otherwise.
    :param probas: ndarray, shape = [n_samples, n_classes], predicted probabilities.
    :param axis_labels: array of strings , shape = [n_classes], labels of the curves.
    :param save_path: string, path to a directory where the figure will saved. If None the figure will not be saved.
    :param show: boolean, if true the figure will be displayed.
    :return: pandas.DataFrame roc_auc_matrix
    """

    # Calculate roc_auc_matrics
    roc_auc_matrics = numpy.ones((probas.shape[1],probas.shape[1]))

    for first in range(probas.shape[1]):
        for second in range(probas.shape[1]):

            if first == second:
                continue

            weights = ((labels[:, first] != 0) + (labels[:, second] != 0)) * 1.

            roc_auc = roc_auc_score(labels[:, first], probas[:, first]/probas[:, second], sample_weight=weights)

            roc_auc_matrics[first, second] = roc_auc


    # Save roc_auc_matrics
    matrix = pandas.DataFrame(columns=['Class'] + axis_labels)
    matrix['Class'] = axis_labels

    for num in range(len(axis_labels)):

        matrix[axis_labels[num]] = roc_auc_matrics[num, :]

    if save_path != None:
        matrix.to_csv(save_path + "/class_vs_class_roc_auc_matrix.csv")


    # Plot roc_auc_matrics
    plt.figure(figsize=(10,7))
    plt.imshow(roc_auc_matrics, interpolation='nearest')
    tick_marks = numpy.arange(len(axis_labels))
    plt.xticks(tick_marks, axis_labels, rotation=45, size=15)
    plt.yticks(tick_marks, axis_labels, size=15)
    plt.clim([0.8,1])
    plt.colorbar()
    plt.tight_layout()
    plt.title('Particle vs particle roc aucs', size=15)

    if save_path != None:
        plt.savefig(save_path + "/overall_roc_auc.png")

    if show == True:
        plt.show()

    plt.clf()
    plt.close()

    return matrix

def get_roc_auc_ration_matrix(matrix_one, matrix_two, save_path=None, show=True):

    """
    Divide matrix_one to matrix_two.
    :param matrix_one: pandas.DataFrame with column 'Class' which contain class names.
    :param matrix_two: pandas.DataFrame with column 'Class' which contain class names.
    :param save_path: string, path to a directory where the figure will saved. If None the figure will not be saved.
    :param show: boolean, if true the figure will be displayed.
    :return: pandas.DataFrame roc_auc_ratio_matrix
    """

    # Calculate roc_auc_matrics
    classes = list(matrix_one.Class.values)
    roc_auc_matrics = numpy.ones((len(classes), len(classes)))

    for first in range(len(classes)):
        for second in range(len(classes)):

            roc_auc_one = matrix_one[classes[second]][matrix_one.Class == classes[first]].values[0]
            roc_auc_two = matrix_two[classes[second]][matrix_two.Class == classes[first]].values[0]

            roc_auc_matrics[first, second] = roc_auc_one / roc_auc_two

    # Save roc_auc_matrics
    matrix = pandas.DataFrame(columns=['Class'] + classes)
    matrix['Class'] = classes

    for num in range(len(classes)):

        matrix[classes[num]] = roc_auc_matrics[num, :]

    if save_path != None:
        matrix.to_csv(save_path + "/class_vs_class_roc_auc_matrix.csv")

    # Plot roc_auc_matrics
    from matplotlib import cm
    plt.figure(figsize=(10,7))
    plt.imshow(roc_auc_matrics, interpolation='nearest', cmap=cm.seismic)
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=15)
    plt.yticks(tick_marks, classes, size=15)
    plt.clim([0.9,1.1])
    plt.colorbar()
    plt.tight_layout()
    plt.title('Particle vs particle roc aucs ratio', size=15)

    if save_path != None:
        plt.savefig(save_path + "/overall_roc_auc.png")

    if show == True:
        plt.show()

    plt.clf()
    plt.close()

    return matrix