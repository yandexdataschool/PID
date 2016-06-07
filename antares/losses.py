import numpy
from scipy.stats import rankdata
from hep_ml.losses import AbstractLossFunction, BinFlatnessLossFunction, AbstractFlatnessLossFunction, HessianLossFunction
from hep_ml.metrics_utils import bin_to_group_indices, compute_bin_indices, compute_group_weights, \
    group_indices_to_groups_matrix
import losses_fortran
from hep_ml.losses import check_sample_weight

class AdaLossFunction(HessianLossFunction):
    """ AdaLossFunction is the same as Exponential Loss Function (aka exploss) """

    def fit(self, X, y, sample_weight):
        self.sample_weight = check_sample_weight(y, sample_weight=sample_weight,
                                                 normalize=True, normalize_by_class=True)
        self.y_signed = numpy.array(2 * y - 1, dtype='float32')
        HessianLossFunction.fit(self, X, y, sample_weight=self.sample_weight)
        return self

    def __call__(self, y_pred):
        return numpy.sum(self.sample_weight * numpy.exp(- self.y_signed * y_pred))

    def negative_gradient(self, y_pred):
        return self.y_signed * self.sample_weight * numpy.exp(- self.y_signed * y_pred)

    def hessian(self, y_pred):
        return self.sample_weight * numpy.exp(- self.y_signed * y_pred)

    def prepare_tree_params(self, y_pred):
        return self.negative_gradient(y_pred), numpy.ones(len(y_pred))
    
    
class BinFlatnessLossFunctionPercentile(AbstractFlatnessLossFunction):
    def __init__(self, uniform_features, uniform_label, n_bins=10, power=2., fl_coefficient=3.,
                 allow_wrong_signs=True):
        """
        This loss function contains separately penalty for non-flatness and for bad prediction quality.
        See [FL]_ for details.

        :math:`\text{loss} =\text{ExpLoss} + c \times \text{FlatnessLoss}`

        FlatnessLoss computed using binning of uniform variables

        :param list[str] uniform_features: names of features, along which we want to obtain uniformity of predictions
        :param int|list[int] uniform_label: the label(s) of classes for which uniformity is desired
        :param int n_bins: number of bins along each variable
        :param float power: the loss contains the difference :math:`| F - F_bin |^p`, where p is power
        :param float fl_coefficient: multiplier for flatness_loss. Controls the tradeoff of quality vs uniformity.
        :param bool allow_wrong_signs: defines whether gradient may different sign from the "sign of class"
            (i.e. may have negative gradient on signal). If False, values will be clipped to zero.

        .. [FL] A. Rogozhnikov et al, New approaches for boosting to uniformity
            http://arxiv.org/abs/1410.4140
        """
        self.n_bins = n_bins
        AbstractFlatnessLossFunction.__init__(self, uniform_features,
                                              uniform_label=uniform_label, power=power,
                                              fl_coefficient=fl_coefficient,
                                              allow_wrong_signs=allow_wrong_signs)

    def _compute_groups_indices(self, X, y, label):
        """Returns a list, each element is events' indices in some group."""
        label_mask = y == label
        extended_bin_limits = []
        for var in self.uniform_features:
            extended_bin_limits.append(numpy.percentile(X[var][label_mask], numpy.linspace(0, 100, 2 * self.n_bins + 1)))
        groups_indices = list()
        for shift in [0, 1]:
            bin_limits = []
            for axis_limits in extended_bin_limits:
                bin_limits.append(axis_limits[1 + shift:-1:2])
            bin_indices = compute_bin_indices(X.ix[:, self.uniform_features].values, bin_limits=bin_limits)
            groups_indices += list(bin_to_group_indices(bin_indices, mask=label_mask))
        return groups_indices
    
    
class SumFlatLossFunction(AbstractLossFunction):
    """
    Compute loss separately for two variables (in our context for track Pt and track P):
    Loss = [Loss(track Pt) + Loss(track P)] / 2
    """
    def __init__(self, p_loss, pt_loss):
        self.pt_loss = pt_loss
        self.p_loss = p_loss
        
    def fit(self, X, y, sample_weight):
        self.pt_loss.fit(X, y, sample_weight)
        self.p_loss.fit(X, y, sample_weight)
        return self

    def __call__(self, y_pred):
        return self.pt_loss(y_pred) + self.p_loss(y_pred)

    def negative_gradient(self, y_pred):
        return (self.pt_loss.negative_gradient(y_pred) + self.p_loss.negative_gradient(y_pred)) / 2.

    def prepare_tree_params(self, y_pred):
        return (self.pt_loss.negative_gradient(y_pred) + self.p_loss.negative_gradient(y_pred)) / 2., numpy.ones(len(y_pred))
    
    
class SumFlatLossFunctionSpeedup(AbstractLossFunction):
    """
    Compute loss separately for two variables (in our context for track Pt and track P):
    Loss = [Loss(track Pt) + Loss(track P)] / 2
    """
    def __init__(self, log_loss, flatness_loss):
        self.log_loss = log_loss
        self.flatness_loss = flatness_loss
        
    def fit(self, X, y, sample_weight):
        self.log_loss.fit(X, y, sample_weight)
        self.flatness_loss.fit(X, y, sample_weight)
        return self

    def __call__(self, y_pred):
        return self.log_loss(y_pred) + self.flatness_loss(y_pred)
    
    def negative_gradient(self, y_pred):
        return self.log_loss.prepare_tree_params(y_pred)[0] + self.flatness_loss.prepare_tree_params(y_pred)[0]

    def prepare_tree_params(self, y_pred):
        return self.log_loss.prepare_tree_params(y_pred)[0] + self.flatness_loss.prepare_tree_params(y_pred)[0], \
    numpy.ones(len(y_pred))
    

class OneDimensionalFlatnessLossFunction(AbstractLossFunction):
    def __init__(self, uniform_features, n_bins=20, uniform_label=1, n_threads=1):
        """
        Only flatness loss, no AdaLoss included!
        Only one feature
        Uniform_features = dictinary{name : coefficient}
        """
        self.uniform_features = uniform_features
        self.n_bins = n_bins
        self.uniform_label = uniform_label
        self.uniform_mask = None
        self.n_threads = n_threads
        
    def __call__(self, predictions):
        return 0.
        
    def fit(self, X, y, sample_weight):
        assert len(X) == len(y) == len(sample_weight), 'different lengths!'
        self.uniform_mask = y == self.uniform_label
        self.sample_weight = sample_weight[self.uniform_mask].astype('float64')
        self.sample_weight /= self.sample_weight.sum()
        self.binning_contributions = []
        self.bin_indices = []
        self.bin_weights = []    
        for uniform_feature, uniform_coeff in self.uniform_features.items():
            
            needed_feature = X[uniform_feature].values[self.uniform_mask].copy()
            all_thresholds = numpy.percentile(needed_feature, numpy.linspace(0, 100, 2 * self.n_bins + 1)[1:-1])
            even_thresholds = all_thresholds[::2]
            odd_thresholds = all_thresholds[1::2]

            for thresholds in [even_thresholds, odd_thresholds]:
                self.binning_contributions.append(uniform_coeff / 2.)
                indices = numpy.searchsorted(thresholds, needed_feature)
                self.bin_indices.append(indices)
                weights = self.sample_weight / numpy.bincount(indices, weights=self.sample_weight)[indices]
                self.bin_weights.append(weights)
                
        self.bin_indices = numpy.array(self.bin_indices, dtype='int32')
        self.bin_weights = numpy.array(self.bin_weights)
        self.passed_trivial_weight = numpy.ones(len(y), dtype='float64')
        return self
        
    def prepare_tree_params_old(self, predictions):
        sorter = numpy.argsort(predictions[self.uniform_mask])
        bins_cumsums = numpy.zeros([len(self.bin_indices), self.n_bins + 1])
        general_cumsum = 0.
        
        uniform_gradients = numpy.zeros(len(sorter))
        
        for event in sorter:
            weight = self.sample_weight[event]
            general_cumsum += weight
            sum_bin_cumsums = 0.
            for i, (bin_indices, bin_weights, binning_contribution) in \
                    enumerate(zip(self.bin_indices, self.bin_weights, self.binning_contributions)):
                bins_cumsums[i, bin_indices[event]] += bin_weights[event]
                sum_bin_cumsums += bins_cumsums[i, bin_indices[event]] * binning_contribution
            
            uniform_gradients[event] = - general_cumsum * sum(self.binning_contributions) + sum_bin_cumsums
        gradients = numpy.zeros(len(predictions)) 
        gradients[self.uniform_mask] = uniform_gradients
        print general_cumsum
        print bins_cumsums 
        return gradients, self.passed_trivial_weight


    def prepare_tree_params(self, predictions):
        """
        works only for trivial weights
        """
        global_order = rankdata(predictions[self.uniform_mask], method='ordinal')
#         numpy.argsort(numpy.argsort())
        
        other_Fs = numpy.zeros(len(global_order), dtype='float64')
        n_binnings = len(self.bin_indices)
        parameter_bins = 2 * len(predictions) // self.n_bins
        for binning in range(n_binnings):
            orders = losses_fortran.compute_bin_orderings(global_order, self.bin_indices[binning], n_bins=self.n_bins + 1, 
                                                          sorter_bins=parameter_bins, n_threads=self.n_threads)
            other_Fs += self.bin_weights[binning] * orders * self.binning_contributions[binning]
            
        gradients = numpy.zeros(len(predictions)) 
        F_global = global_order * (1. / len(global_order) * numpy.sum(self.binning_contributions) )

        gradients[self.uniform_mask] = - F_global + other_Fs
        return gradients, self.passed_trivial_weight