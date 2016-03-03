import numpy
import pandas
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



def labels_transform(labels):

    """
    Transform labels from shape = [n_samples] to shape = [n_samples, n_classes]
    :param labels: array
    :return: ndarray, transformed labels
    """

    classes = numpy.unique(labels)

    new_labels = numpy.zeros((len(labels), len(classes)))
    for cl in classes:
        new_labels[:, cl] = (labels == cl) * 1.

    return new_labels


def get_roc_curves(labels, probas, curve_labels, save_path=None, show=True):
    """
    Creates roc curve for each class vs rest.
    :param labels: array, shape = [n_samples], labels for the each class.
    1 - if a sample belongs to the class, 0 - otherwise.
    :param probas: ndarray, shape = [n_samples, n_classes], predicted probabilities.
    :param curve_labels: array of strings , shape = [n_classes], labels of the curves.
    :param save_path: string, path to a directory where the figure will saved. If None the figure will not be saved.
    :param show: boolean, if true the figure will be displayed.
    """

    labels = labels_transform(labels)

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
    :param labels: array, shape = [n_samples], labels for the each class.
    1 - if a sample belongs to the class, 0 - otherwise.
    :param probas: ndarray, shape = [n_samples, n_classes], predicted probabilities.
    :param axis_labels: array of strings , shape = [n_classes], labels of the curves.
    :param save_path: string, path to a directory where the figure will saved. If None the figure will not be saved.
    :param show: boolean, if true the figure will be displayed.
    :return: pandas.DataFrame roc_auc_matrix
    """

    labels = labels_transform(labels)

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

def get_flatness_threshold(n_simulations, q, track):

    """
    Compute percentile of CvM test for flatness
    :param n_simulations: int number of simulations
    :param q: percentile
    :param track: array, variable along which the CvM test computes
    :return: float
    """

    cvm_pdf = []

    for step in range(n_simulations):

        proba_rand = numpy.random.random(len(track))
        cvm_pdf.append(compute_cvm(proba_rand, track))

    cvm_pdf = numpy.array(cvm_pdf)
    threshold = numpy.percentile(cvm_pdf, q)

    return threshold

def get_flatness_table(data, labels, probas, class_names, save_path=None):

    """
    Compute CvM tests for TrackP and TrackPt for each classes.
    :param data: pandas.DataFrame, data
    :param labels: array, shape = [n_samples], labels for the each class.
    1 - if a sample belongs to the class, 0 - otherwise.
    :param probas: ndarray, shape = [n_samples, n_classes], predicted probabilities.
    :param axis_labels: array of strings , shape = [n_classes], labels of the curves.
    :param class_names: string, path to a directory where the figure will saved. If None the figure will not be saved.
    :return: flatness pandas.DataFrame
    """

    labels = labels_transform(labels)

    GeV = 1000
    limits = {"TrackP": [100*GeV, 0],
              "TrackPt": [10*GeV, 0] }

    track_p = data.TrackP.values
    sel_p = (track_p >= limits["TrackP"][1]) * (track_p < limits["TrackP"][0])

    track_pt = data.TrackPt.values
    sel_pt = (track_pt >= limits["TrackPt"][1]) * (track_pt < limits["TrackPt"][0])



    cvm_track_p = []
    cvm_track_pt = []

    threshold_track_p = []
    threshold_track_pt = []

    for num in range(probas.shape[1]):

        sel_class_p = sel_p * (labels[:, num] == 1)
        sel_class_pt = sel_pt * (labels[:, num] == 1)

        cvm_p = compute_cvm(probas[sel_class_p, num], track_p[sel_class_p])
        cvm_track_p.append(cvm_p)

        threshold_p = get_flatness_threshold(100, 95, track_p[sel_class_p])
        threshold_track_p.append(threshold_p)


        cvm_pt = compute_cvm(probas[sel_class_pt, num], track_pt[sel_class_pt])
        cvm_track_pt.append(cvm_pt)

        threshold_pt = get_flatness_threshold(100, 95, track_pt[sel_class_pt])
        threshold_track_pt.append(threshold_pt)

    flatness = pandas.DataFrame(columns=['Class', 'TrackP', 'TrackPt'])
    flatness['Class'] = class_names
    flatness['TrackP'] = cvm_track_p
    flatness['TrackPt'] = cvm_track_pt
    flatness['P_Conf_level'] = threshold_track_p
    flatness['Pt_Conf_level'] = threshold_track_pt

    if save_path != None:
        flatness.to_csv(save_path + "/flatness.csv")

    return flatness

def get_flatness_ratio(flatness_one, flatness_two, save_path=None):

    """
    Get ratio of flatness_one and flatness_two
    :param flatness_one: pandas.DataFrame with column 'Class' which contain class names.
    :param flatness_two: pandas.DataFrame with column 'Class' which contain class names.
    :param save_path: string, path to a directory where the figure will saved. If None the figure will not be saved.
    :return: pandas.DataFrame
    """

    classes = flatness_one.Class.values

    flatness_arr = numpy.zeros((len(classes), 2))

    for num in range(len(classes)):

        flat_one = flatness_one[flatness_one.Class == classes[num]][[u'TrackP', u'TrackPt']].values
        flat_two = flatness_two[flatness_two.Class == classes[num]][[u'TrackP', u'TrackPt']].values

        flatness_arr[num, :] = flat_one / flat_two


    flatness = pandas.DataFrame(columns=['Class', 'TrackP', 'TrackPt'])
    flatness['Class'] = classes
    flatness['TrackP'] = flatness_arr[:, 0]
    flatness['TrackPt'] = flatness_arr[:, 1]

    if save_path != None:
        flatness.to_csv(save_path + "/rel_flatness.csv")

    return flatness

from rep.utils import get_efficiencies
from rep.plotting import ErrorPlot

def flatness_p_figure(proba, track_p, track_name, particle_name, save_path=None, show=False):

    """
    Plot signal efficiency vs TrackP figure.
    :param proba: array, shape = [n_samples], predicted probabilities.
    :param track_p: array, shape = [n_samples], TrackP values.
    :param track_name: string, name.
    :param particle_name: string, name.
    :param save_path: string, path to a directory where the figure will saved. If None the figure will not be saved.
    :param show: boolean, if true the figure will be displayed.
    """

    thresholds = numpy.percentile(proba, 100 - numpy.array([5, 20, 40, 60, 80]))

    eff = get_efficiencies(proba,
                           track_p,
                           bins_number=30,
                           errors=True,
                           ignored_sideband=0,
                           thresholds=thresholds)

    for i in thresholds:
        eff[i] = (eff[i][0], 100. * eff[i][1], 100. * eff[i][2], eff[i][3])

    plot_fig = ErrorPlot(eff)
    plot_fig.ylim = (0, 100)

    plot_fig.plot(new_plot=True, figsize=(10,7))
    plt.legend(['Eff = 5 %', 'Eff = 20 %', 'Eff = 40 %', 'Eff = 60 %', 'Eff = 80 %'], loc='best',prop={'size':15})
    plt.xlabel(track_name + ' ' + particle_name + ' Momentum / MeV/c', size=15)
    plt.xticks(size=15)
    plt.ylabel('Efficiency / %', size=15)
    plt.yticks(size=15)
    plt.title('Flatness_SignalMVAEffVTrackP_' + track_name + ' ' + particle_name, size=15)

    if save_path != None:
        plt.savefig(save_path + "/" + 'Flatness_SignalMVAEffVTrackP_' + track_name + '_' + particle_name + ".png")

    if show == True:
        plt.show()

    plt.clf()
    plt.close()


def flatness_pt_figure(proba, track_pt, track_name, particle_name, save_path=None, show=False):

    """
    Plot signal efficiency vs TrackPt figure.
    :param proba: array, shape = [n_samples], predicted probabilities.
    :param track_p: array, shape = [n_samples], TrackPt values.
    :param track_name: string, name.
    :param particle_name: string, name.
    :param save_path: string, path to a directory where the figure will saved. If None the figure will not be saved.
    :param show: boolean, if true the figure will be displayed.
    """

    thresholds = numpy.percentile(proba, 100 - numpy.array([5, 20, 40, 60, 80]))

    eff = get_efficiencies(proba,
                           track_pt,
                           bins_number=30,
                           errors=True,
                           ignored_sideband=0,
                           thresholds=thresholds)

    for i in thresholds:
        eff[i] = (eff[i][0], 100. * eff[i][1], 100. * eff[i][2], eff[i][3])

    plot_fig = ErrorPlot(eff)
    plot_fig.ylim = (0, 100)

    plot_fig.plot(new_plot=True, figsize=(10,7))
    plt.legend(['Eff = 5 %', 'Eff = 20 %', 'Eff = 40 %', 'Eff = 60 %', 'Eff = 80 %'], loc='best',prop={'size':15})
    plt.xlabel(track_name + ' ' + particle_name + ' Transverse Momentum / MeV/c', size=15)
    plt.xticks(size=15)
    plt.ylabel('Efficiency / %', size=15)
    plt.yticks(size=15)
    plt.title('Flatness_SignalMVAEffVTrackPt_' + track_name + ' ' + particle_name, size=15)

    if save_path != None:
        plt.savefig(save_path + "/" + 'Flatness_SignalMVAEffVTrackPt_' + track_name + '_' + particle_name + ".png")

    if show == True:
        plt.show()

    plt.clf()
    plt.close()


def get_all_flatness_figures(data, probas, labels, track_name, particle_names, save_path=None, show=False):

    """
    Plot signal efficiency vs TrackP figure.
    :param data: pandas.dataFrame() data.
    :param probas: bdarray, shape = [n_samples, n_classes], predicted probabilities.
    :param labels: array, shape = [n_samples], class labels
    :param track_p: array, shape = [n_samples], TrackP values.
    :param track_name: string, name.
    :param particle_names: list of strings, particle names.
    :param save_path: string, path to a directory where the figure will saved. If None the figure will not be saved.
    :param show: boolean, if true the figure will be displayed.
    """

    labels = labels_transform(labels)

    GeV = 1000
    limits = {"TrackP": [100*GeV, 0],
              "TrackPt": [10*GeV, 0] }

    track_p = data.TrackP.values
    sel_p = (track_p >= limits["TrackP"][1]) * (track_p < limits["TrackP"][0])

    track_pt = data.TrackPt.values
    sel_pt = (track_pt >= limits["TrackPt"][1]) * (track_pt < limits["TrackPt"][0])


    for num in range(len(particle_names)):

        sel_class_p = sel_p * (labels[:, num] == 1)
        sel_class_pt = sel_pt * (labels[:, num] == 1)

        probas[sel_class_p, num], track_p[sel_class_p]

        flatness_p_figure(probas[sel_class_p, num],
                          track_p[sel_class_p],
                          track_name,
                          particle_names[num],
                          save_path=save_path,
                          show=show)

        flatness_pt_figure(probas[sel_class_pt, num],
                          track_pt[sel_class_pt],
                          track_name,
                          particle_names[num],
                          save_path=save_path,
                          show=show)