__author__ = 'mikhail91'

import numpy
import pandas
import matplotlib.pyplot as plt
import matplotlib as mpl


particle_pdg_codes = {"Ghost": 0, "Electron": 11, "Muon": 13, "Pion": 211, "Kaon": 321, "Proton": 2212}


n_cuts_mva = 80
n_cuts_dll = 150

n_bins = 20
n_bins_2d = 50


mva_out_cuts = [0, 0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999]
baseline_cuts = mva_out_cuts

particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
colors = ['k', 'r', 'g', 'b', 'gold', 'm']


def get_bins(x, y, bins, x_min, x_max):

    step = 1. * ( x_max - x_min ) / bins
    edges = [x_min + i * step for i in range(0, bins+1)]

    y_means = []
    y_err = []
    x_err = []
    x_means = []

    for i in range(0, len(edges)-1):

        left = edges[i]
        right = edges[i+1]

        y_bin = y[(x >= left) * (x < right)]

        y_means.append(y_bin.mean())
        y_err.append(1. * y_bin.std() / (len(y_bin) + 0.001))
        x_means.append(0.5*(left + right))
        x_err.append(0.5*(-left + right))

    return x_means, y_means, x_err, y_err

def get_hist(var, bins, min_var, max_var):


    step = 1. * (max_var - min_var) / bins
    edges = [min_var + i * step for i in range(0, bins + 1)]

    var_bins = []
    var_errs = []
    n_bins = []
    n_errs = []

    for i in range(0, len(edges) - 1):

        left = edges[i]
        right = edges[i+1]

        var_bins.append(0.5 * (right + left))
        var_errs.append(0.5 * (right - left))

        n = 1. * len(var[(var >= left) * (var < right)])
        n_bins.append(n / (len(var) + 0.001))
        n_errs.append(numpy.sqrt(n)/ (len(var) + 0.001))

    var_bins = numpy.array(var_bins)
    n_bins = numpy.array(n_bins)
    var_errs = numpy.array(var_errs)
    n_errs = numpy.array(n_errs)

    return var_bins, n_bins, var_errs, n_errs

def poisson_error(sel, total):

    p = 1. * sel / total
    n = 1. - p
    N = total

    err = numpy.sqrt(numpy.abs(p * n / (N)))

    return 100 * err

def get_eff(mva, bins, min_mva, max_mva):

    step = 1. * (max_mva - min_mva) / bins
    edges = [min_mva + i * step for i in range(0, bins + 1)]

    effs = []
    eff_errs = []
    mva_errs = []

    for edge in edges:

        one_eff = 100. * len(mva[mva >= edge]) / (len(mva) + 0.001)
        effs.append(one_eff)

        one_eff_err = poisson_error(len(mva[mva >= edge]), len(mva) + 0.001)
        eff_errs.append(one_eff_err)

        mva_errs.append(0.5 * step)

    return effs, edges, eff_errs, mva_errs


def get_eff_v_var(mva, mva_cut, var, bins, min_var, max_var):

    # step = 1. * (max_var - min_var) / bins
    # edges = [min_var + i * step for i in range(0, bins + 1)]

    var_copy = var.copy()
    var_copy.sort(axis=0)
    N = len(var_copy) / bins
    edges = [var_copy[i * N] for i in range(0, bins)] + [var_copy[-1]]

    effs = []
    vars_bins = []
    eff_errs = []
    var_errs = []

    for i in range(0, len(edges)-1):

        left = edges[i]
        right = edges[i + 1]

        bin_mva = mva[(var >= left) * (var < right)]

        one_eff = 100. * len(bin_mva[bin_mva >= mva_cut]) / (len(bin_mva) + 0.001)
        effs.append(one_eff)

        one_eff_err = poisson_error(len(bin_mva[bin_mva >= mva_cut]), len(bin_mva) + 0.001)
        eff_errs.append(one_eff_err)

        vars_bins.append(0.5 * (right + left))
        var_errs.append(0.5 * (right - left))

    return effs, vars_bins, eff_errs, var_errs

def get_por_eff(mva, labels, bins, min_mva, max_mva):

    step = 1. * (max_mva - min_mva) / bins
    edges = [min_mva + i * step for i in range(0, bins + 1)]

    effs = []
    eff_errs = []
    purs = []
    pur_errs = []
    mva_errs = []

    for edge in edges:

        one_eff = 100. * len(mva[mva >= edge]) / (len(mva) + 0.001)
        effs.append(one_eff)

        one_eff_err = poisson_error(len(mva[mva >= edge]), len(mva) + 0.001)
        eff_errs.append(100.*one_eff_err)

        one_pur = 100. * len(labels[(labels==1) * (mva >= edge)]) / (len(labels[mva >= edge]) + 0.001)
        purs.append(one_pur)

        one_pur_err = poisson_error(len(labels[(labels==1) * (mva >= edge)]), len(labels[mva >= edge]) + 0.001)
        pur_errs.append(100.*one_eff_err)

        mva_errs.append(0.5 * step)

    return effs, eff_errs, purs, pur_errs


def get_miss_and_eff(mva_p_one, mva_p_two, bins, min_mva, max_mva):

    edges = numpy.percentile(mva_p_two, 100 - numpy.array(numpy.arange(0, 100.1, 0.1)))

    effs_p_one = []
    eff_errs_p_one = []
    effs_p_two = []
    eff_errs_p_two = []

    for edge in edges:

        one_eff_p_one = 100. * len(mva_p_one[mva_p_one >= edge]) / (len(mva_p_one) + 0.001)
        effs_p_one.append(one_eff_p_one)

        one_eff_err_p_one = poisson_error(len(mva_p_one[mva_p_one >= edge]), len(mva_p_one) + 0.001)
        eff_errs_p_one.append(one_eff_err_p_one)

        one_eff_p_two = 100. * len(mva_p_two[mva_p_two >= edge]) / (len(mva_p_two) + 0.001)
        effs_p_two.append(one_eff_p_two)

        one_eff_err_p_two = poisson_error(len(mva_p_two[mva_p_two >= edge]), len(mva_p_two) + 0.001)
        eff_errs_p_two.append(one_eff_err_p_two)

    return effs_p_one, effs_p_two, eff_errs_p_one, eff_errs_p_two


def get_pur(mva, labels, bins, min_mva, max_mva):

    step = 1. * (max_mva - min_mva) / bins
    edges = [min_mva + i * step for i in range(0, bins + 1)]

    purs = []
    pur_errs = []
    mva_errs = []
    mva_bins = []

    for edge in edges:

        one_pur = 100. * len(labels[(labels==1) * (mva >= edge)]) / (len(labels[mva >= edge]) + 0.001)
        if len(labels[mva >= edge]) == 0:
            one_pur = 100
        purs.append(one_pur)

        one_pur_err = poisson_error(len(labels[(labels==1) * (mva >= edge)]), len(labels[mva >= edge]) + 0.001)
        pur_errs.append(one_pur_err)

        mva_errs.append(0.5 * step)

    return purs, pur_errs, edges, mva_errs







# MVA inputs

def Inputs(params, eval_data, eval_proba, eval_labels, features, log=False, path="pic"):

    import os

    if not os.path.exists(path + "/pdf/Inputs"):
        os.makedirs(path + "/pdf/Inputs")

    if not os.path.exists(path + "/png/Inputs"):
        os.makedirs(path + "/png/Inputs")


    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']


    for var in features:

        var_data2 = eval_data[var].values
        var_data = var_data2[var_data2 != -999]

        p_types = numpy.abs(eval_data[u'MCParticleType'].values)
        p_types = p_types[var_data2 != -999]

        plt.figure(figsize=(10,7))

        ymax = 0.
        ymin= 1.

        for particle, color in zip(particles, colors):

            pdg_code = particle_pdg_codes[particle]
            hist_data = var_data[p_types == pdg_code]

            var_bins, n_in_bins, var_errs, n_errs = get_hist(hist_data, n_bins,
                                                      var_data.min(), var_data.max())

            plt.errorbar(var_bins, n_in_bins, xerr=var_errs, yerr=n_errs, fmt='none', ecolor=color)
            plt.scatter(var_bins, n_in_bins, c=color, label=particle)

            plt.ylabel("", size=15)
            plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " " + var, size=15)
            plt.grid(True, lw = 2, ls = ':', c = '.75')
            plt.xlim(var_data.min(), var_data.max())


            if (n_in_bins+n_errs).max() > ymax:
                ymax=(n_in_bins+n_errs).max()
            if len(n_in_bins[n_in_bins > 0]):
                if n_in_bins[n_in_bins > 0].min() < ymin:
                    ymin = n_in_bins[n_in_bins > 0].min()

            plt.ylim(ymin=0, ymax=ymax)
            plt.legend(loc='best', prop={'size':15})
            plt.xticks(size=15)
            plt.yticks(size=15)

            if log:
                if ymin >= ymax:
                    ymax = ymin
                plt.ylim(ymin, ymax)
                plt.yscale('log', nonposy='clip')

        if not log:
            plt.title("Inputs_" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny", size=15)
            plt.savefig(path + "/pdf/Inputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny.pdf")
            plt.savefig(path + "/png/Inputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny.png")
        else:
            plt.title("Inputs_" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Logy", size=15)
            plt.savefig(path + "/pdf/Inputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Logy.pdf")
            plt.savefig(path + "/png/Inputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Logy.png")

        plt.clf()
        plt.close('all')
    return 1

# MVA output V input profiles

def MVAVInputs(params, eval_data, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf/MVAVInputs"):
        os.makedirs(path + "/pdf/MVAVInputs")

    if not os.path.exists(path + "/png/MVAVInputs"):
        os.makedirs(path + "/png/MVAVInputs")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    for var in features:

        plt.figure(figsize=(10,7))

        var_data2 = eval_data[var].values
        var_data = var_data2[var_data2 != -999]
        p_types = numpy.abs(eval_data[u'MCParticleType'].values)
        p_types = p_types[var_data2 != -999]

        for particle, color in zip(particles, colors):

            pdg_code = particle_pdg_codes[particle]

            pdg_data = var_data[p_types == pdg_code]
            pdg_proba = (eval_proba[var_data2 != -999])[p_types == pdg_code][:, 1]


            x_min = var_data.min()
            x_max = var_data.max()

            if var == 'TrackP':
                x_min = 0
                x_max = 100000
            elif var == 'TrackPt':
                x_min = 0
                x_max = 10000

            x_means, y_means, x_err, y_err = get_bins(pdg_data, pdg_proba, n_bins, x_min, x_max)

            plt.errorbar(x_means, y_means, xerr=x_err, yerr=y_err, fmt='none', ecolor=color)
            plt.scatter(x_means, y_means, c=color, label=particle)

            plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " " + var, size=15)
            plt.ylabel(params['TRACK'] + " " + params['PARTICLE'] + " " + "MVA Output", size=15)
            plt.ylim(0, 1.1)
            plt.plot([x_min, x_max], [1,1], 'k--', linewidth=1)
            plt.xlim(x_min, x_max)
            plt.yticks(numpy.arange(0, 1.01, 0.1), size=15)
            plt.xticks(size=15)
            plt.grid(True, lw = 2, ls = ':', c = '.75')
            plt.legend(loc='best', prop={'size':15})

            plt.title("MVAVInputs_" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny", size=15)

        plt.savefig(path + "/pdf/MVAVInputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny.pdf")
        plt.savefig(path + "/png/MVAVInputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny.png")

        plt.clf()
        plt.close('all')

    return 1

# Baseline MVA output V input profiles
def BaselineMVAVInputs(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf/BaselineMVAVInputs"):
        os.makedirs(path + "/pdf/BaselineMVAVInputs")

    if not os.path.exists(path + "/png/BaselineMVAVInputs"):
        os.makedirs(path + "/png/BaselineMVAVInputs")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    for var in features:

        var_data2 = eval_data[var].values
        var_data = var_data2[var_data2 != -999]
        p_types = numpy.abs(eval_data[u'MCParticleType'].values)
        p_types = p_types[var_data2 != -999]
        baseline_mva = eval_proba_baseline[params['PARTICLE']].values
        baseline_mva = baseline_mva[var_data2 != -999]

        plt.figure(figsize=(10,7))

        for particle, color in zip(particles, colors):

            pdg_code = particle_pdg_codes[particle]

            pdg_data = var_data[p_types == pdg_code]
            pdg_comb = baseline_mva[p_types == pdg_code]

            x_min = var_data.min()
            x_max = var_data.max()

            if var == 'TrackP':
                x_min = 0
                x_max = 100000
            elif var == 'TrackPt':
                x_min = 0
                x_max = 10000

            x_means, y_means, x_err, y_err = get_bins(pdg_data, pdg_comb, n_bins, x_min, x_max)

            plt.errorbar(x_means, y_means, xerr=x_err, yerr=y_err, fmt='none', ecolor=color)
            plt.scatter(x_means, y_means, c=color, label=particle)


            plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " " + var)
            plt.ylabel(params['TRACK'] + " " + params['PARTICLE'] + " " + 'Baseline MVA' + " Output")
            plt.ylim(0, 1)
            plt.xlim(x_min, x_max)
            plt.grid(True, lw = 2, ls = ':', c = '.75')
            plt.legend(loc='best', prop={'size':15})
            plt.xticks(size=15)
            plt.yticks(size=15)

            plt.title("BaselineMVAVInputs_" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny", size=15)


        plt.savefig(path + "/pdf/BaselineMVAVInputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny.pdf")
        plt.savefig(path + "/png/BaselineMVAVInputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny.png")

        plt.clf()
        plt.close('all')

    return 1


# Eff vs MVA Out | BaselineMVA cuts

def MVAEffForBaselineMVACut(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf/MVAEffForBaselineMVACut"):
        os.makedirs(path + "/pdf/MVAEffForBaselineMVACut")

    if not os.path.exists(path + "/png/MVAEffForBaselineMVACut"):
        os.makedirs(path + "/png/MVAEffForBaselineMVACut")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    baseline_mva = eval_proba_baseline[params['PARTICLE']].values


    for cut in baseline_cuts:

        cut_data = eval_data[baseline_mva > cut]
        cut_p_types = cut_data.MCParticleType.values
        cut_mva_out = eval_proba[baseline_mva > cut][:, 1]

        plt.figure(figsize=(10,7))

        for particle, color in zip(particles, colors):

            pdg_code = particle_pdg_codes[particle]

            pdg_data = cut_data[cut_p_types == pdg_code]
            pdg_mva = cut_mva_out[cut_p_types == pdg_code]

            effs, edges, eff_errs, mva_errs = get_eff(pdg_mva, n_bins, 0, 1)

            plt.errorbar(edges, effs, xerr=mva_errs, yerr=eff_errs, fmt='none', ecolor=color)
            plt.scatter(edges, effs, c=color, label=particle)

            plt.ylabel("Efficiency / %", size=15)
            plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " MVA cut value", size=15)

            plt.xlim(0, 1)
            plt.xticks(numpy.arange(0, 1.01, 0.1), size=15)
            plt.ylim(0, 110)
            plt.plot([0, 1], [100,100], 'k--', linewidth=1)
            plt.yticks(numpy.arange(0, 111, 10), size=15)

            plt.grid(True, lw = 2, ls = ':', c = '.75')

            plt.legend(loc='best', prop={'size':15})
            plt.title("MVAEffForBaselineMVACut_" + params['TRACK'] + "_" + params['PARTICLE'] + "-MVAEff-BaselineMVACut%.3f" % cut, size=15)

        plt.savefig(path + "/pdf/MVAEffForBaselineMVACut" + "/" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-MVAEff-BaselineMVACut%.3f.pdf" % cut)
        plt.savefig(path + "/png/MVAEffForBaselineMVACut" + "/" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-MVAEff-BaselineMVACut%.3f.png" % cut)

        plt.clf()
        plt.close('all')

    return 1


# Eff vs BaselineMVA Out | MVA cuts
def BaselineMVAEffForMVACut(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf/BaselineMVAEffForMVACut"):
        os.makedirs(path + "/pdf/BaselineMVAEffForMVACut")

    if not os.path.exists(path + "/png/BaselineMVAEffForMVACut"):
        os.makedirs(path + "/png/BaselineMVAEffForMVACut")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    baseline_mva = eval_proba_baseline[params['PARTICLE']].values


    for mva_cut in mva_out_cuts:

        cut_data = eval_data[eval_proba[:, 1] > mva_cut]
        cut_p_types = cut_data.MCParticleType.values
        cut_baseline_mva = baseline_mva[eval_proba[:, 1] > mva_cut]

        plt.figure(figsize=(10,7))

        for particle, color in zip(particles, colors):

            pdg_code = particle_pdg_codes[particle]

            pdg_data = cut_data[cut_p_types == pdg_code]
            pdg_baseline_mva = cut_baseline_mva[cut_p_types == pdg_code]

            effs, edges, eff_errs, baseline_mva_errs = get_eff(pdg_baseline_mva, n_bins,
                                                           0, 1)

            plt.errorbar(edges, effs, xerr=baseline_mva_errs, yerr=eff_errs, fmt='none', ecolor=color,
                         label=particle, linestyle='None')
            #plt.scatter(edges, effs, c=color, label=particle)

            plt.ylabel("Efficiency / %", size=15)
            plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " BaselineMVA cut value", size=15)

            plt.xlim(0, 1)
            plt.ylim(0, 110)
            plt.plot(0, 1, [100,100], 'k--', linewidth=1)
            plt.yticks(numpy.arange(0, 111, 10), size=15)
            plt.xticks(size=15)

            plt.grid(True, lw = 2, ls = ':', c = '.75')

            plt.legend(loc='best', prop={'size':15})
            plt.title("BaselineMVAEffForMVACut_" + params['TRACK'] + "_" + params['PARTICLE'] + "-BaselineMVAEff-MVACut%.4f" % mva_cut, size=15)


        plt.savefig(path + "/pdf/BaselineMVAEffForMVACut" + "/" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-BaselineMVAEff-MVACut%.4f.pdf" % mva_cut)
        plt.savefig(path + "/png/BaselineMVAEffForMVACut" + "/" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-BaselineMVAEff-MVACut%.4f.png" % mva_cut)

        plt.clf()
        plt.close('all')

    return 1


# Eff vs TrackP | BaselineMVA cuts

def BaselineMVAEffVTrackP(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf/BaselineMVAEffVTrackP"):
        os.makedirs(path + "/pdf/BaselineMVAEffVTrackP")

    if not os.path.exists(path + "/png/BaselineMVAEffVTrackP"):
        os.makedirs(path + "/png/BaselineMVAEffVTrackP")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    baseline_mva = eval_proba_baseline[params['PARTICLE']].values


    for cut in baseline_cuts:

        cut_p_types = eval_data[eval_data.TrackP.values <= 100000].MCParticleType.values
        cut_track_p = eval_data[eval_data.TrackP.values <= 100000]['TrackP'].values
        cut_baseline_mva = baseline_mva[eval_data.TrackP.values <= 100000]

        plt.figure(figsize=(10,7))

        for particle, color in zip(particles, colors):

            pdg_code = particle_pdg_codes[particle]

            pdg_track_p = cut_track_p[cut_p_types == pdg_code]
            pdg_baseline_mva = cut_baseline_mva[cut_p_types == pdg_code]

            effs, edges, eff_errs, var_errs = get_eff_v_var(pdg_baseline_mva, cut, pdg_track_p,
                                                           n_bins, cut_track_p.min(), cut_track_p.max())

            plt.errorbar(edges, effs, xerr=var_errs, yerr=eff_errs, fmt='none', ecolor=color)
            plt.scatter(edges, effs, c=color, label=particle)

            plt.ylabel("Efficiency / %")
            plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " Momentum / MeV/c")

            plt.xlim(0, 100000)
            plt.ylim(0, 110)
            plt.plot([0, 100000], [100,100], 'k--', linewidth=1)

            plt.yticks(numpy.arange(0, 111, 10), size=15)
            plt.xticks(size=15)

            plt.grid(True, lw = 2, ls = ':', c = '.75')

            plt.legend(loc='best', prop={'size':15})
            plt.title("BaselineMVAEffVTrackP_" + params['TRACK'] + "_" + params['PARTICLE'] + "-BaselineMVACut%.4f" % cut, size=15)

        plt.savefig(path + "/pdf/BaselineMVAEffVTrackP" + "/BaselineMVAEffVTrackP_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-BaselineMVACut%.4f.pdf" % cut)
        plt.savefig(path + "/png/BaselineMVAEffVTrackP" + "/BaselineMVAEffVTrackP_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-BaselineMVACut%.4f.png" % cut)

        plt.clf()
        plt.close('all')

    return 1


# Eff vs TrackPt | BaselineMVA cuts

def BaselineMVAEffVTrackPt(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf/BaselineMVAEffVTrackPt"):
        os.makedirs(path + "/pdf/BaselineMVAEffVTrackPt")

    if not os.path.exists(path + "/png/BaselineMVAEffVTrackPt"):
        os.makedirs(path + "/png/BaselineMVAEffVTrackPt")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    baseline_mva = eval_proba_baseline[params['PARTICLE']].values


    for cut in baseline_cuts:

        cut_p_types = numpy.abs(eval_data[eval_data.TrackPt.values <= 10000].MCParticleType.values)
        cut_track_pt = eval_data[eval_data.TrackPt.values <= 10000]['TrackPt'].values
        cut_baseline_mva = baseline_mva[eval_data.TrackPt.values <= 10000]

        plt.figure(figsize=(10,7))

        for particle, color in zip(particles, colors):

            pdg_code = particle_pdg_codes[particle]

            pdg_track_pt = cut_track_pt[cut_p_types == pdg_code]
            pdg_baseline_mva = cut_baseline_mva[cut_p_types == pdg_code]

            effs, edges, eff_errs, var_errs = get_eff_v_var(pdg_baseline_mva, cut, pdg_track_pt,
                                                           n_bins, cut_track_pt.min(), cut_track_pt.max())

            plt.errorbar(edges, effs, xerr=var_errs, yerr=eff_errs, fmt='none', ecolor=color)
            plt.scatter(edges, effs, c=color, label=particle)

            plt.ylabel("Efficiency / %", size=15)
            plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " Transverse Momentum / MeV/c", size=15)

            plt.ylim(0,110)
            plt.xlim(0, 10000)
            plt.plot([0, 10000], [100,100], 'k--', linewidth=1)
            plt.yticks(numpy.arange(0, 111, 10), size=15)
            plt.xticks(size=15)

            plt.grid(True, lw = 2, ls = ':', c = '.75')

            plt.legend(loc='best', prop={'size':15})
            plt.title("BaselineMVAEffVTrackPt_" + params['TRACK'] + "_" + params['PARTICLE'] + "-BaselineMVACut%.4f" % cut, size=15)

        plt.savefig(path + "/pdf/BaselineMVAEffVTrackPt" + "/BaselineMVAEffVTrackPt_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-BaselineMVACut%.4f.pdf" % cut)
        plt.savefig(path + "/png/BaselineMVAEffVTrackPt" + "/BaselineMVAEffVTrackPt_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-BaselineMVACut%.4f.png" % cut)
        plt.clf()
        plt.close('all')

    return 1


# Eff vs TrackP | MVA cuts
def MVAEffVTrackP(params, eval_data, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf/MVAEffVTrackP"):
        os.makedirs(path + "/pdf/MVAEffVTrackP")

    if not os.path.exists(path + "/png/MVAEffVTrackP"):
        os.makedirs(path + "/png/MVAEffVTrackP")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']



    for mva_cut in mva_out_cuts:

        cut_p_types = eval_data[eval_data.TrackP.values <= 100000].MCParticleType.values
        cut_track_p = eval_data[eval_data.TrackP.values <= 100000]['TrackP'].values
        cut_mva = eval_proba[eval_data.TrackP.values <= 100000, 1]

        plt.figure(figsize=(10,7))

        for particle, color in zip(particles, colors):

            pdg_code = particle_pdg_codes[particle]

            pdg_track_p = cut_track_p[cut_p_types == pdg_code]
            pdg_mva = cut_mva[cut_p_types == pdg_code]

            effs, edges, eff_errs, var_errs = get_eff_v_var(pdg_mva, mva_cut, pdg_track_p,
                                                           n_bins, cut_track_p.min(), cut_track_p.max())

            plt.errorbar(edges, effs, xerr=var_errs, yerr=eff_errs, fmt='none', ecolor=color)
            plt.scatter(edges, effs, c=color, label=particle)

            plt.ylabel("Efficiency / %", size=15)
            plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " Momentum / MeV/c", size=15)

            plt.xlim(0, 100000)
            plt.ylim(0,110)
            plt.plot([0, 100000], [100,100], 'k--', linewidth=1)
            plt.yticks(numpy.arange(0, 111, 10), size=15)
            plt.xticks(size=15)

            plt.grid(True, lw = 2, ls = ':', c = '.75')

            plt.legend(loc='best', prop={'size':15})
            plt.title("MVAEffVTrackP_" + params['TRACK'] + "_" + params['PARTICLE'] + "-MVACut%.4f" % mva_cut, size=15)

        plt.savefig(path + "/pdf/MVAEffVTrackP" + "/MVAEffVTrackP_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-MVACut%.4f.pdf" % mva_cut)
        plt.savefig(path + "/png/MVAEffVTrackP" + "/MVAEffVTrackP_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-MVACut%.4f.png" % mva_cut)

        plt.clf()
        plt.close('all')

    return 1


# Eff vs TrackPt | MVA cuts

def MVAEffVTrackPt(params, eval_data, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf/MVAEffVTrackPt"):
        os.makedirs(path + "/pdf/MVAEffVTrackPt")

    if not os.path.exists(path + "/png/MVAEffVTrackPt"):
        os.makedirs(path + "/png/MVAEffVTrackPt")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']



    for mva_cut in mva_out_cuts:

        cut_p_types = numpy.abs(eval_data[eval_data.TrackPt.values <= 10000].MCParticleType.values)
        cut_track_pt = eval_data[eval_data.TrackPt.values <= 10000]['TrackPt'].values
        cut_mva = eval_proba[eval_data.TrackPt.values <= 10000, 1]

        plt.figure(figsize=(10,7))

        for particle, color in zip(particles, colors):

            pdg_code = particle_pdg_codes[particle]

            pdg_track_pt = cut_track_pt[cut_p_types == pdg_code]
            pdg_mva = cut_mva[cut_p_types == pdg_code]

            effs, edges, eff_errs, var_errs = get_eff_v_var(pdg_mva, mva_cut, pdg_track_pt,
                                                           n_bins, cut_track_pt.min(), cut_track_pt.max())

            plt.errorbar(edges, effs, xerr=var_errs, yerr=eff_errs, fmt='none', ecolor=color)
            plt.scatter(edges, effs, c=color, label=particle)

            plt.ylabel("Efficiency / %", size=15)
            plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " Transverse Momentum / MeV/c", size=15)

            plt.xlim(0, 10000)
            plt.ylim(0, 110)
            plt.plot([0, 10000], [100,100], 'k--', linewidth=1)
            plt.yticks(numpy.arange(0, 111, 10), size=15)
            plt.xticks(size=15)

            plt.grid(True, lw = 2, ls = ':', c = '.75')

            plt.legend(loc='best', prop={'size':15})
            plt.title("MVAEffVTrackPt_" + params['TRACK'] + "_" + params['PARTICLE'] + "-MVACut%.4f" % mva_cut, size=15)

        plt.savefig(path + "/pdf/MVAEffVTrackPt" + "/MVAEffVTrackPt_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-MVACut%.4f.pdf" % mva_cut)
        plt.savefig(path + "/png/MVAEffVTrackPt" + "/MVAEffVTrackPt_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-MVACut%.4f.png" % mva_cut)

        plt.clf()
        plt.close('all')
    return 1


# BaselineMVA
def BaselineMVA(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, log=False, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    baseline_mva = eval_proba_baseline[params['PARTICLE']].values

    plt.figure(figsize=(10,7))

    baseline_mva_data = baseline_mva
    cut_p_types = numpy.abs(eval_data.MCParticleType.values)

    ymax = 0.
    ymin = 1.

    for particle, color in zip(particles, colors):

        pdg_code = particle_pdg_codes[particle]
        pdg_baseline_mva = baseline_mva_data[cut_p_types == pdg_code]

        var_bins, n_in_bins, var_errs, n_errs = get_hist(pdg_baseline_mva, n_bins,
                                                      0, 1)

        plt.errorbar(var_bins, n_in_bins, xerr=var_errs, yerr=n_errs, fmt='none', ecolor=color)
        plt.scatter(var_bins, n_in_bins, c=color, label=particle)

        if (n_in_bins+n_errs).max() > ymax:
            ymax=(n_in_bins+n_errs).max()
        if len(n_in_bins[n_in_bins > 0]) >0:
            if n_in_bins[n_in_bins > 0].min() < ymin:
                ymin = n_in_bins[n_in_bins > 0].min()

        plt.ylabel("")
        plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " " + 'BaselineMVA', size=15)

        plt.xlim(0, 1)
        plt.ylim(0, ymax)
        plt.xticks(size=15)
        plt.yticks(size=15)

        if log:
            if ymax <= ymin:
                ymax = ymin
            plt.ylim(ymin, ymax)
            plt.yscale('log', nonposy='clip')

        plt.grid(True, lw = 2, ls = ':', c = '.75')

        plt.legend(loc='best', prop={'size':15})

    if not log:
        plt.title(params['TRACK'] + "_" + params['PARTICLE'] + 'BaselineMVA' + "_Liny", size=15)
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + 'BaselineMVA' + "_Liny.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + 'BaselineMVA' + "_Liny.png")
    else:
        plt.title(params['TRACK'] + "_" + params['PARTICLE'] + 'BaselineMVA' + "_Logy", size=15)
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + 'BaselineMVA' + "_Logy.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + 'BaselineMVA' + "_Logy.png")

    plt.clf()
    plt.close('all')
    return 1


# MVA Out
def MVAOut(params, eval_data, eval_proba, eval_labels, features, log=False, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    plt.figure(figsize=(10,7))

    mva_data = eval_proba[:,1]
    cut_p_types = numpy.abs(eval_data.MCParticleType.values)

    ymax = 0.
    ymin = 1.

    for particle, color in zip(particles, colors):

        pdg_code = particle_pdg_codes[particle]

        pdg_proba = mva_data[cut_p_types == pdg_code]

        var_bins, n_in_bins, var_errs, n_errs = get_hist(pdg_proba, n_bins, 0, 1)

        plt.errorbar(var_bins, n_in_bins, xerr=var_errs, yerr=n_errs, fmt='none', ecolor=color)
        plt.scatter(var_bins, n_in_bins, c=color, label=particle)

        if (n_in_bins+n_errs).max() > ymax:
            ymax=(n_in_bins+n_errs).max()
        if len(n_in_bins[n_in_bins > 0]) >0:
            if n_in_bins[n_in_bins > 0].min() < ymin:
                ymin = n_in_bins[n_in_bins > 0].min()

        plt.xlim(0, 1)
        plt.xticks(numpy.arange(0, 1.01, .10), size=15)
        plt.ylim(0, ymax)
        plt.yticks(size=15)

        if log:
            if ymax <= ymin:
                ymax = ymin
            plt.ylim(ymin, ymax)
            plt.yscale('log', nonposy='clip')

            plt.grid(True, lw = 2, ls = ':', c = '.75')

        plt.ylabel("")
        plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " MVA Output", size=15)
        plt.legend(loc='best', prop={'size':15})

    if not log:
        plt.title(params['TRACK'] + "_" + params['PARTICLE'] + "MVAOut" + "_Liny", size=15)
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "MVAOut" + "_Liny.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "MVAOut" + "_Liny.png")
    else:
        plt.title(params['TRACK'] + "_" + params['PARTICLE'] + "MVAOut" + "_Logy", size=15)
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "MVAOut" + "_Logy.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "MVAOut" + "_Logy.png")

    plt.clf()
    plt.close('all')
    return 1

# Purity V mva
# Purity V BaselineMVA

def EffPurity(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, log=False, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    plt.figure(figsize=(10,7))
    baseline_mva = eval_proba_baseline[params['PARTICLE']].values

    effs, eff_errs, purs, pur_errs = get_por_eff(eval_proba[:, 1], eval_labels, 1000, 0, 1)
    effs_d, eff_errs_d, purs_d, pur_errs_d = get_por_eff(baseline_mva, eval_labels, 100,
                                                 0, 1)

    plt.plot(effs, purs, label='MVA Out', color='b', linewidth=2)
    plt.plot(effs_d, purs_d, label='BaselineMVA Out', color='r', linewidth=2)
    #plt.scatter(effs, purs, )
    #plt.errorbar(effs, purs, xerr=eff_errs, yerr=pur_errs, fmt='none')

    plt.ylabel("Purity / %", size=15)
    plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " Efficiency / %", size=15)

    plt.xticks(numpy.arange(0, 101, 10), size=15)
    plt.yticks(numpy.arange(0, 111, 10), size=15)
    plt.xlim(0, 100)
    plt.ylim(0, 110)
    plt.plot([0, 100], [100,100], 'k--', linewidth=1)
    plt.grid(True, lw = 2, ls = ':', c = '.75')

    plt.legend(loc='best', prop={'size':15})

    plt.title(params['TRACK'] + "_" + params['PARTICLE'] + "-IDEff_V_Electron-Purity", size=15)
    plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-IDEff_V_Electron-Purity.pdf")
    plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-IDEff_V_Electron-Purity.png")

    plt.clf()
    plt.close('all')
    return 1


# Eff MVA V MisEff

def EffMissIDEff(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    p_types = numpy.abs(eval_data.MCParticleType.values)
    baseline_mva = eval_proba_baseline[params['PARTICLE']].values

    for particle, color in zip(particles, colors):

        plt.figure(figsize=(10,7))

        pdg_code_one = particle_pdg_codes[particle]
        pdg_mva_one = eval_proba[p_types == pdg_code_one, 1]
        pdg_baseline_mva_one = (baseline_mva)[p_types == pdg_code_one]

        if len(pdg_baseline_mva_one) == 0:
            continue

        pdg_code_two = particle_pdg_codes[params['PARTICLE']]
        pdg_mva_two = eval_proba[p_types == pdg_code_two, 1]
        pdg_baseline_mva_two = (baseline_mva)[p_types == pdg_code_two]

        effs_p_one, effs_p_two, eff_errs_p_one, eff_errs_p_two = \
        get_miss_and_eff(pdg_mva_one, pdg_mva_two, 1000, 0, 1)

        effs_p_one2, effs_p_two2, eff_errs_p_one2, eff_errs_p_two2 = \
        get_miss_and_eff(pdg_baseline_mva_one, pdg_baseline_mva_two, 100, 0, 1)

        #plt.errorbar(effs_p_two, effs_p_one, xerr=eff_errs_p_two, yerr=eff_errs_p_one, fmt='none', ecolor=color)
        plt.plot(effs_p_two, effs_p_one, c='b', label='MVA Out', linewidth=2)
        plt.plot(effs_p_two2, effs_p_one2, c='r', label='BaselineMVA Out', linewidth=2)

        plt.ylabel(particle + " MisID Efficiency / %", size=15)
        plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " Efficiency / %", size=15)
        plt.legend(loc='best', prop={'size':15})

        plt.xticks(numpy.arange(0, 101, 10), size=15)
        plt.xlim(0, 100)
        #plt.plot([0, 100000], [100,100], color='k', linewidth=2)

        plt.yscale('log', nonposy='clip')
        plt.ylim(0, 110)
        plt.yticks(size=15)
        plt.plot([0, 100], [100,100], 'k--', linewidth=1)
        plt.grid(True, lw = 2, ls = ':', c = '.75')
        plt.title(params['TRACK'] + "_" + params['PARTICLE'] + "-IDEff_V_" + particle + "-MisIDEff", size=15)


        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                    "-IDEff_V_" + particle + "-MisIDEff.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                    "-IDEff_V_" + particle + "-MisIDEff.png")


        plt.clf()
        plt.close('all')

    return 1

# Eff MVA V MisEff

def EffOverallMissIDEff(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, log=False, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    p_types = numpy.abs(eval_data.MCParticleType.values)
    baseline_mva = eval_proba_baseline[params['PARTICLE']].values

    plt.figure(figsize=(10,7))

    pdg_code_one = particle_pdg_codes[params['PARTICLE']]
    pdg_mva_one = eval_proba[p_types != pdg_code_one, 1]
    pdg_baseline_mva_one = (baseline_mva)[p_types != pdg_code_one]

    pdg_code_two = particle_pdg_codes[params['PARTICLE']]
    pdg_mva_two = eval_proba[p_types == pdg_code_two, 1]
    pdg_baseline_mva_two = (baseline_mva)[p_types == pdg_code_two]

    effs_p_one, effs_p_two, eff_errs_p_one, eff_errs_p_two = \
    get_miss_and_eff(pdg_mva_one, pdg_mva_two, 1000, 0, 1)

    effs_p_one2, effs_p_two2, eff_errs_p_one2, eff_errs_p_two2 = \
    get_miss_and_eff(pdg_baseline_mva_one, pdg_baseline_mva_two, 100, 0, 1)

    #plt.errorbar(effs_p_two, effs_p_one, xerr=eff_errs_p_two, yerr=eff_errs_p_one, fmt='none', ecolor=color)
    plt.plot(effs_p_two, effs_p_one, c='b', label='MVA Out', linewidth=2)
    plt.plot(effs_p_two2, effs_p_one2, c='r', label='BaselineMVA Out', linewidth=2)

    plt.ylabel("Overall MisID Efficiency / %", size=15)
    plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " Efficiency / %", size=15)
    plt.legend(loc='best', prop={'size':15})

    plt.xticks(numpy.arange(0, 101, 10), size=15)
    plt.xlim(0, 100)
    plt.ylim(0, 110)
    plt.yticks(size=15)
    plt.plot([0, 100], [100,100], 'k--', linewidth=1)

    plt.grid(True, lw = 2, ls = ':', c = '.75')

    if log:
        plt.yscale('log', nonposy='clip')
        plt.ylim(0, 110)
        plt.plot([0, 100], [100,100], 'k--', linewidth=1)

    if log:
        plt.title(params['TRACK'] + "_" + params['PARTICLE'] + "-IDEff_V_" + "OverallMisIDEff_Logy", size=15)
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                    "-IDEff_V_" + "OverallMisIDEff_Logy.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                    "-IDEff_V_" + "OverallMisIDEff_Logy.png")
    else:
        plt.title(params['TRACK'] + "_" + params['PARTICLE'] + "-IDEff_V_" + "OverallMisIDEff_Liny", size=15)
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                    "-IDEff_V_" + "OverallMisIDEff_Liny.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                    "-IDEff_V_" + "OverallMisIDEff_Liny.png")


    plt.clf()
    plt.close('all')

    return 1

def MVABaselineMVA(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, log=False, signal=1, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    p_types = numpy.abs(eval_data.MCParticleType.values)
    baseline_mva = eval_proba_baseline[params['PARTICLE']].values

    plt.figure(figsize=(10,7))

    baseline_mva_data2 = baseline_mva[eval_labels==signal]
    baseline_mva_data = baseline_mva_data2[(baseline_mva_data2 >= 0) * (baseline_mva_data2 < 1)]
    mva_data = eval_proba[eval_labels==signal, 1]
    mva_data = mva_data[(baseline_mva_data2 >= 0) * (baseline_mva_data2 < 1)]

    if log:
        plt.hist2d(baseline_mva_data, mva_data, bins=n_bins_2d, norm = mpl.colors.LogNorm())
    else:
        plt.hist2d(baseline_mva_data, mva_data, bins=n_bins_2d)

    plt.colorbar()

    plt.ylim(0, 1)
    plt.yticks(size=15)
    plt.xlim(0, 1)
    plt.xticks(size=15)

    plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " " + 'BaselineMVA')
    plt.ylabel(params['TRACK'] + " " + params['PARTICLE'] + " MVA Output")

    if log and signal:
        plt.title(params['TRACK'] + "_" + params['PARTICLE'] + "-MVAOutVBaselineMVA-Signal_Logz", size=15)
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                    "-MVAOutVBaselineMVA-Signal_Logz.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                    "-MVAOutVBaselineMVA-Signal_Logz.png")
    elif log and not signal:
        plt.title(params['TRACK'] + "_" + params['PARTICLE'] + "-MVAOutVBaselineMVA-Background_Logz", size=15)
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                    "-MVAOutVBaselineMVA-Background_Logz.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                    "-MVAOutVBaselineMVA-Background_Logz.png")
    elif not log and signal:
        plt.title(params['TRACK'] + "_" + params['PARTICLE'] + "-MVAOutVBaselineMVA-Signal_Linz", size=15)
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                    "-MVAOutVBaselineMVA-Signal_Linz.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                    "-MVAOutVBaselineMVA-Signal_Linz.png")
    else:
        plt.title(params['TRACK'] + "_" + params['PARTICLE'] + "-MVAOutVBaselineMVA-Background_Linz", size=15)
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                    "-MVAOutVBaselineMVA-Background_Linz.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                    "-MVAOutVBaselineMVA-Background_Linz.png")
    plt.clf()
    plt.close('all')


    return 1

# Pur V MVA
def PurityVMVAOut(params, eval_data, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    plt.figure(figsize=(10,7))

    purs, pur_errs, edges, mva_errs = get_pur(eval_proba[:, 1], eval_labels, n_bins, 0, 1)

    plt.scatter(edges, purs)
    plt.errorbar(edges, purs, xerr=mva_errs, yerr=pur_errs, fmt='none')

    plt.ylim(0,110)
    plt.xlim(0, 1)
    plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " MVA Output", size=15)
    plt.ylabel('Purity / %', size=15)

    plt.xticks(numpy.arange(0, 1.01, .10), size=15)
    plt.yticks(numpy.arange(0, 111, 10), size=15)
    plt.plot([0, 1.0], [100,100], 'k--', linewidth=1)

    plt.grid(True, lw = 2, ls = ':', c = '.75')
    plt.title(params['TRACK'] + "_" + params['PARTICLE'] + "-PurityVMVAOut", size=15)

    plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                "-PurityVMVAOut.pdf")
    plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                "-PurityVMVAOut.png")

    plt.clf()
    plt.close('all')

    return 1

# Pur V MVA
def PurityVBaselineMVA(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    plt.figure(figsize=(10,7))

    baseline_mva = eval_proba_baseline[params['PARTICLE']]

    purs, pur_errs, edges, mva_errs = get_pur(baseline_mva, eval_labels, n_bins,
                                          0, 1)

    plt.scatter(edges, purs)
    plt.errorbar(edges, purs, xerr=mva_errs, yerr=pur_errs, fmt='none')

    plt.ylim(0,110)
    plt.xlim(0, 1)
    plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " " + 'BaselineMVA', size=15)
    plt.ylabel('Purity / %', size=15)

    plt.yticks(numpy.arange(0, 111, 10), size=15)
    plt.xticks(size=15)
    plt.plot([0, 1], [100,100], 'k--', linewidth=1)

    plt.grid(True, lw = 2, ls = ':', c = '.75')
    plt.title(params['TRACK'] + "_" + params['PARTICLE'] + "-PurityVBaselineMVA", size=15)

    plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                "-PurityVBaselineMVA.pdf")
    plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + \
                "-PurityVBaselineMVA.png")

    plt.clf()
    plt.close('all')

    return 1

# MAIN function

def all_figures(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path="pic"):

    Inputs(params, eval_data, eval_proba, eval_labels, features, log=False, path=path)
    Inputs(params, eval_data, eval_proba, eval_labels, features, log=True, path=path)

    MVAVInputs(params, eval_data, eval_proba, eval_labels, features, path=path)
    BaselineMVAVInputs(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path=path)

    MVAEffForBaselineMVACut(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path=path)
    BaselineMVAEffForMVACut(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path=path)

    BaselineMVAEffVTrackP(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path=path)
    BaselineMVAEffVTrackPt(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path=path)

    MVAEffVTrackP(params, eval_data, eval_proba, eval_labels, features, path=path)
    MVAEffVTrackPt(params, eval_data, eval_proba, eval_labels, features, path=path)

    BaselineMVA(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, log=False, path=path)
    BaselineMVA(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, log=True, path=path)

    MVAOut(params, eval_data, eval_proba, eval_labels, features, log=False, path=path)
    MVAOut(params, eval_data, eval_proba, eval_labels, features, log=True, path=path)

    EffPurity(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, log=False, path=path)

    EffMissIDEff(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path=path)

    EffOverallMissIDEff(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, log=False, path=path)
    EffOverallMissIDEff(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, log=True, path=path)

    MVABaselineMVA(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, log=False, signal=1, path=path)
    MVABaselineMVA(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, log=False, signal=0, path=path)
    MVABaselineMVA(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, log=True, signal=1, path=path)
    MVABaselineMVA(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, log=True, signal=0, path=path)

    PurityVMVAOut(params, eval_data, eval_proba, eval_labels, features, path=path)
    PurityVBaselineMVA(params, eval_data, eval_proba_baseline, eval_proba, eval_labels, features, path=path)

    return 1


