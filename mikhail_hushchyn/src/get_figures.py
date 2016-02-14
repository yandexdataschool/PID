__author__ = 'mikhail91'

import numpy
import pandas
import matplotlib.pyplot as plt
import matplotlib as mpl

dll_min = {'Electron': -15., 'Muon': -15., 'Pion': -100., 'Kaon': -50., 'Proton': -50, 'Ghost': 0.}
dll_max = {'Electron': 15., 'Muon': 15., 'Pion': 100., 'Kaon': 50., 'Proton': 50, 'Ghost': 1.}
comb_dlls = {'Electron': "CombDLLe", 'Muon': "CombDLLmu", 'Pion': "CombDLLpi",
             'Kaon': "CombDLLk", 'Proton': "CombDLLp", 'Ghost': "TrackGhostProbability"}
particle_pdg_codes = {"Ghost": 0, "Electron": 11, "Muon": 13, "Pion": 211, "Kaon": 321, "Proton": 2212}


n_cuts_mva = 80
n_cuts_dll = 150

n_bins = 100
n_bins_2d = 50

#mvastep = 1. / (n_cuts_mva - 1)
#dllstep = ( dll_max[params['PARTICLE']] - dll_min[params['PARTICLE']] ) / (n_cuts_dll-1)

mva_out_cuts = [0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999]
dll_cuts = [-8.0, -7.0, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
            0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]

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

    step = 1. * (max_var - min_var) / bins
    edges = [min_var + i * step for i in range(0, bins + 1)]

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
        var_errs.append(0.5 * step)

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

    step = 1. * (max_mva - min_mva) / bins
    edges = [min_mva + i * step for i in range(0, bins + 1)]

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
            plt.legend(loc='best')

            if log:
                if ymin >= ymax:
                    ymax = ymin
                plt.ylim(ymin, ymax)
                plt.yscale('log', nonposy='clip')

            if not log:
                plt.savefig(path + "/pdf/Inputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny.pdf")
                plt.savefig(path + "/png/Inputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny.png")
            else:
                plt.savefig(path + "/pdf/Inputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Logy.pdf")
                plt.savefig(path + "/png/Inputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Logy.png")

        plt.close()
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
            plt.ylim(0, 1)
            plt.xlim(x_min, x_max)
            plt.yticks(numpy.arange(0, 1.01, 0.1))
            plt.grid(True, lw = 2, ls = ':', c = '.75')
            plt.legend(loc='best')

        plt.savefig(path + "/pdf/MVAVInputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny.pdf")
        plt.savefig(path + "/png/MVAVInputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny.png")

        plt.close()

    return 1

# DLL output V input profiles
def DLLVInputs(params, eval_data, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf/DLLVInputs"):
        os.makedirs(path + "/pdf/DLLVInputs")

    if not os.path.exists(path + "/png/DLLVInputs"):
        os.makedirs(path + "/png/DLLVInputs")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    for var in features:

        var_data2 = eval_data[var].values
        var_data = var_data2[var_data2 != -999]
        p_types = numpy.abs(eval_data[u'MCParticleType'].values)
        p_types = p_types[var_data2 != -999]
        dll = eval_data[comb_dlls[params['PARTICLE']]].values
        dll = dll[var_data2 != -999]

        plt.figure(figsize=(10,7))

        for particle, color in zip(particles, colors):

            pdg_code = particle_pdg_codes[particle]

            pdg_data = var_data[p_types == pdg_code]
            pdg_comb = dll[p_types == pdg_code]

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
            plt.ylabel(params['TRACK'] + " " + params['PARTICLE'] + " " + comb_dlls[params['PARTICLE']] + " Output")
            plt.ylim(dll_min[params['PARTICLE']], dll_max[params['PARTICLE']])
            plt.xlim(x_min, x_max)
            plt.grid(True, lw = 2, ls = ':', c = '.75')
            plt.legend(loc='best')


        plt.savefig(path + "/pdf/DLLVInputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny.pdf")
        plt.savefig(path + "/png/DLLVInputs" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + "-" + var + "_Liny.png")


        plt.close()

    return 1


# Eff vs MVA Out | CombDDL cuts

def MVAEffForDLLCut(params, eval_data, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf/MVAEffForDLLCut"):
        os.makedirs(path + "/pdf/MVAEffForDLLCut")

    if not os.path.exists(path + "/png/MVAEffForDLLCut"):
        os.makedirs(path + "/png/MVAEffForDLLCut")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    comb_dll = comb_dlls[params['PARTICLE']]


    for dll_cut in dll_cuts:

        cut_data = eval_data[eval_data[comb_dll].values > dll_cut]
        cut_p_types = cut_data.MCParticleType.values
        cut_mva_out = eval_proba[eval_data[comb_dll].values > dll_cut][:, 1]

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
            plt.xticks(numpy.arange(0, 1.01, 0.1))
            plt.ylim(0, 100)
            plt.yticks(numpy.arange(0, 101, 10))

            plt.grid(True, lw = 2, ls = ':', c = '.75')

            plt.legend(loc='best')

        plt.savefig(path + "/pdf/MVAEffForDLLCut" + "/" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-MVAEff-DLLCut%.3f.pdf" % dll_cut)
        plt.savefig(path + "/png/MVAEffForDLLCut" + "/" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-MVAEff-DLLCut%.3f.png" % dll_cut)


        plt.close()

    return 1


# Eff vs DLL Out | MVA cuts
def DLLEffForMVACut(params, eval_data, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf/DLLEffForMVACut"):
        os.makedirs(path + "/pdf/DLLEffForMVACut")

    if not os.path.exists(path + "/png/DLLEffForMVACut"):
        os.makedirs(path + "/png/DLLEffForMVACut")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    comb_dll = comb_dlls[params['PARTICLE']]


    for mva_cut in mva_out_cuts:

        cut_data = eval_data[eval_proba[:, 1] > mva_cut]
        cut_p_types = cut_data.MCParticleType.values
        cut_comb_dll = cut_data[comb_dll].values

        plt.figure(figsize=(10,7))

        for particle, color in zip(particles, colors):

            pdg_code = particle_pdg_codes[particle]

            pdg_data = cut_data[cut_p_types == pdg_code]
            pdg_comb_dll = cut_comb_dll[cut_p_types == pdg_code]

            effs, edges, eff_errs, comb_dll_errs = get_eff(pdg_comb_dll, n_bins,
                                                           dll_min[params['PARTICLE']], dll_max[params['PARTICLE']])

            plt.errorbar(edges, effs, xerr=comb_dll_errs, yerr=eff_errs, fmt='none', ecolor=color,
                         label=particle, linestyle='None')
            #plt.scatter(edges, effs, c=color, label=particle)

            plt.ylabel("Efficiency / %", size=15)
            plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " CombDLL cut value", size=15)

            plt.xlim(dll_min[params['PARTICLE']], dll_max[params['PARTICLE']])
            plt.ylim(0, 100)
            plt.yticks(numpy.arange(0, 101, 10))

            plt.grid(True, lw = 2, ls = ':', c = '.75')

            plt.legend(loc='best')


        plt.savefig(path + "/pdf/DLLEffForMVACut" + "/" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-DLLEff-MVACut%.4f.pdf" % mva_cut)
        plt.savefig(path + "/png/DLLEffForMVACut" + "/" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-DLLEff-MVACut%.4f.png" % mva_cut)

        plt.close()

    return 1


# Eff vs TrackP | ComboDLL cuts

def DLLEffVTrackP(params, eval_data, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf/DLLEffVTrackP"):
        os.makedirs(path + "/pdf/DLLEffVTrackP")

    if not os.path.exists(path + "/png/DLLEffVTrackP"):
        os.makedirs(path + "/png/DLLEffVTrackP")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    comb_dll = comb_dlls[params['PARTICLE']]


    for dll_cut in dll_cuts:

        cut_p_types = eval_data[eval_data.TrackP.values <= 100000].MCParticleType.values
        cut_track_p = eval_data[eval_data.TrackP.values <= 100000]['TrackP'].values
        cut_combo_dll = eval_data[eval_data.TrackP.values <= 100000][comb_dll].values

        plt.figure(figsize=(10,7))

        for particle, color in zip(particles, colors):

            pdg_code = particle_pdg_codes[particle]

            pdg_track_p = cut_track_p[cut_p_types == pdg_code]
            pdg_combo_dll = cut_combo_dll[cut_p_types == pdg_code]

            effs, edges, eff_errs, var_errs = get_eff_v_var(pdg_combo_dll, dll_cut, pdg_track_p,
                                                           n_bins, cut_track_p.min(), cut_track_p.max())

            plt.errorbar(edges, effs, xerr=var_errs, yerr=eff_errs, fmt='none', ecolor=color)
            plt.scatter(edges, effs, c=color, label=particle)

            plt.ylabel("Efficiency / %")
            plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " Momentum / MeV/c")

            plt.xlim(0, 100000)
            plt.ylim(0, 100)
            plt.yticks(numpy.arange(0, 101, 10))

            plt.grid(True, lw = 2, ls = ':', c = '.75')

            plt.legend(loc='best')

        plt.savefig(path + "/pdf/DLLEffVTrackP" + "/DLLEffVTrackP_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-DLLCut%.4f.pdf" % dll_cut)
        plt.savefig(path + "/png/DLLEffVTrackP" + "/DLLEffVTrackP_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-DLLCut%.4f.png" % dll_cut)

        plt.close()

    return 1


# Eff vs TrackPt | ComboDLL cuts

def DLLEffVTrackPt(params, eval_data, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf/DLLEffVTrackPt"):
        os.makedirs(path + "/pdf/DLLEffVTrackPt")

    if not os.path.exists(path + "/png/DLLEffVTrackPt"):
        os.makedirs(path + "/png/DLLEffVTrackPt")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    comb_dll = comb_dlls[params['PARTICLE']]


    for dll_cut in dll_cuts:

        cut_p_types = numpy.abs(eval_data[eval_data.TrackPt.values <= 10000].MCParticleType.values)
        cut_track_pt = eval_data[eval_data.TrackPt.values <= 10000]['TrackPt'].values
        cut_combo_dll = eval_data[eval_data.TrackPt.values <= 10000][comb_dll].values

        plt.figure(figsize=(10,7))

        for particle, color in zip(particles, colors):

            pdg_code = particle_pdg_codes[particle]

            pdg_track_pt = cut_track_pt[cut_p_types == pdg_code]
            pdg_combo_dll = cut_combo_dll[cut_p_types == pdg_code]

            effs, edges, eff_errs, var_errs = get_eff_v_var(pdg_combo_dll, dll_cut, pdg_track_pt,
                                                           n_bins, cut_track_pt.min(), cut_track_pt.max())

            plt.errorbar(edges, effs, xerr=var_errs, yerr=eff_errs, fmt='none', ecolor=color)
            plt.scatter(edges, effs, c=color, label=particle)

            plt.ylabel("Efficiency / %", size=15)
            plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " Transverse Momentum / MeV/c", size=15)

            plt.ylim(0,100)
            plt.xlim(0, 10000)
            plt.yticks(numpy.arange(0, 101, 10))

            plt.grid(True, lw = 2, ls = ':', c = '.75')

            plt.legend(loc='best')

        plt.savefig(path + "/pdf/DLLEffVTrackPt" + "/DLLEffVTrackPt_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-DLLCut%.4f.pdf" % dll_cut)
        plt.savefig(path + "/png/DLLEffVTrackPt" + "/DLLEffVTrackPt_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-DLLCut%.4f.png" % dll_cut)

        plt.close()

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

    comb_dll = comb_dlls[params['PARTICLE']]


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
            plt.ylim(0,100)
            plt.yticks(numpy.arange(0, 101, 10))

            plt.grid(True, lw = 2, ls = ':', c = '.75')

            plt.legend(loc='best')

        plt.savefig(path + "/pdf/MVAEffVTrackP" + "/MVAEffVTrackP_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-MVACut%.4f.pdf" % mva_cut)
        plt.savefig(path + "/png/MVAEffVTrackP" + "/MVAEffVTrackP_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-MVACut%.4f.png" % mva_cut)

        plt.close()

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

    comb_dll = comb_dlls[params['PARTICLE']]


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
            plt.ylim(0, 100)
            plt.yticks(numpy.arange(0, 101, 10))

            plt.grid(True, lw = 2, ls = ':', c = '.75')

            plt.legend(loc='best')

        plt.savefig(path + "/pdf/MVAEffVTrackPt" + "/MVAEffVTrackPt_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-MVACut%.4f.pdf" % mva_cut)
        plt.savefig(path + "/png/MVAEffVTrackPt" + "/MVAEffVTrackPt_" + params['TRACK'] + "_" + \
                    params['PARTICLE'] + "-MVACut%.4f.png" % mva_cut)

        plt.close()
    return 1


# CombDLL
def CombDLL(params, eval_data, eval_proba, eval_labels, features, log=False, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    comb_dll = comb_dlls[params['PARTICLE']]

    plt.figure(figsize=(10,7))

    comb_dll_data = eval_data[comb_dll].values
    cut_p_types = numpy.abs(eval_data.MCParticleType.values)

    ymax = 0.
    ymin = 1.

    for particle, color in zip(particles, colors):

        pdg_code = particle_pdg_codes[particle]
        pdg_comb_dll = comb_dll_data[cut_p_types == pdg_code]

        var_bins, n_in_bins, var_errs, n_errs = get_hist(pdg_comb_dll, n_bins,
                                                      dll_min[params['PARTICLE']], dll_max[params['PARTICLE']])

        plt.errorbar(var_bins, n_in_bins, xerr=var_errs, yerr=n_errs, fmt='none', ecolor=color)
        plt.scatter(var_bins, n_in_bins, c=color, label=particle)

        if (n_in_bins+n_errs).max() > ymax:
            ymax=(n_in_bins+n_errs).max()
        if len(n_in_bins[n_in_bins > 0]) >0:
            if n_in_bins[n_in_bins > 0].min() < ymin:
                ymin = n_in_bins[n_in_bins > 0].min()

        plt.ylabel("")
        plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " " + comb_dll, size=15)

        plt.xlim(dll_min[params['PARTICLE']], dll_max[params['PARTICLE']])
        plt.ylim(0, ymax)

        if log:
            if ymax <= ymin:
                ymax = ymin
            plt.ylim(ymin, ymax)
            plt.yscale('log', nonposy='clip')

        plt.grid(True, lw = 2, ls = ':', c = '.75')

        plt.legend(loc='best')

    if not log:
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + comb_dll + "_Liny.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + comb_dll + "_Liny.png")
    else:
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + comb_dll + "_Logy.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + comb_dll + "_Logy.png")

    plt.close()
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

    comb_dll = comb_dlls[params['PARTICLE']]

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
        plt.xticks(numpy.arange(0, 1.01, .10))
        plt.ylim(0, ymax)

        if log:
            if ymax <= ymin:
                ymax = ymin
            plt.ylim(ymin, ymax)
            plt.yscale('log', nonposy='clip')

            plt.grid(True, lw = 2, ls = ':', c = '.75')

        plt.ylabel("")
        plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " MVA Output", size=15)
        plt.legend(loc='best')

    if not log:
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + "MVAOut" + "_Liny.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + "MVAOut" + "_Liny.png")
    else:
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + "MVAOut" + "_Logy.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + "MVAOut" + "_Logy.png")

    plt.close()
    return 1

# Purity V mva
# Purity V CombDLL

def EffPurity(params, eval_data, eval_proba, eval_labels, features, log=False, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    plt.figure(figsize=(10,7))
    comb_dll = comb_dlls[params['PARTICLE']]

    effs, eff_errs, purs, pur_errs = get_por_eff(eval_proba[:, 1], eval_labels, 10*n_bins, 0, 1)
    effs_d, eff_errs_d, purs_d, pur_errs_d = get_por_eff(eval_data[comb_dll].values, eval_labels, n_bins,
                                                 dll_min[params['PARTICLE']], dll_max[params['PARTICLE']])

    plt.plot(effs, purs, label='MVA Out', color='b', linewidth=2)
    plt.plot(effs_d, purs_d, label=comb_dll, color='r', linewidth=2)
    #plt.scatter(effs, purs, )
    #plt.errorbar(effs, purs, xerr=eff_errs, yerr=pur_errs, fmt='none')

    plt.ylabel("Purity / %", size=15)
    plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " Efficiency / %", size=15)

    plt.xticks(numpy.arange(0, 101, 10))
    plt.yticks(numpy.arange(0, 101, 10))
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, lw = 2, ls = ':', c = '.75')

    plt.legend(loc='best')

    plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + "-IDEff_V_Electron-Purity.pdf")
    plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + "-IDEff_V_Electron-Purity.png")

    plt.close()
    return 1


# Eff MVA V MisEff

def EffMissIDEff(params, eval_data, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    p_types = numpy.abs(eval_data.MCParticleType.values)
    comb_dll = comb_dlls[params['PARTICLE']]

    for particle, color in zip(particles, colors):

        plt.figure(figsize=(10,7))

        pdg_code_one = particle_pdg_codes[particle]
        pdg_mva_one = eval_proba[p_types == pdg_code_one, 1]
        pdg_dll_one = (eval_data[comb_dll])[p_types == pdg_code_one].values

        if len(pdg_dll_one) == 0:
            continue

        pdg_code_two = particle_pdg_codes[params['PARTICLE']]
        pdg_mva_two = eval_proba[p_types == pdg_code_two, 1]
        pdg_dll_two = (eval_data[comb_dll])[p_types == pdg_code_two].values

        effs_p_one, effs_p_two, eff_errs_p_one, eff_errs_p_two = \
        get_miss_and_eff(pdg_mva_one, pdg_mva_two, n_bins, 0, 1)

        effs_p_one2, effs_p_two2, eff_errs_p_one2, eff_errs_p_two2 = \
        get_miss_and_eff(pdg_dll_one, pdg_dll_two, n_bins, dll_min[params['PARTICLE']], dll_max[params['PARTICLE']])

        #plt.errorbar(effs_p_two, effs_p_one, xerr=eff_errs_p_two, yerr=eff_errs_p_one, fmt='none', ecolor=color)
        plt.plot(effs_p_two, effs_p_one, c='b', label='MVA Out', linewidth=2)
        plt.plot(effs_p_two2, effs_p_one2, c='r', label=comb_dll, linewidth=2)

        plt.ylabel(particle + " MissID Efficiency / %", size=15)
        plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " Efficiency / %", size=15)
        plt.legend(loc='best')

        plt.xticks(numpy.arange(0, 101, 10))
        plt.xlim(0, 100)

        plt.yscale('log', nonposy='clip')
        plt.grid(True, lw = 2, ls = ':', c = '.75')

        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                    "-IDEff_V_" + particle + "-MissIDEff.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                    "-IDEff_V_" + particle + "-MissIDEff.png")


        plt.close()

    return 1

# Eff MVA V MisEff

def EffOverallMissIDEff(params, eval_data, eval_proba, eval_labels, features, log=False, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    particles = ["Ghost", "Electron", "Muon", "Pion", "Kaon", "Proton"]
    colors = ['k', 'r', 'g', 'b', 'gold', 'm']

    p_types = numpy.abs(eval_data.MCParticleType.values)
    comb_dll = comb_dlls[params['PARTICLE']]

    plt.figure(figsize=(10,7))

    pdg_code_one = particle_pdg_codes[params['PARTICLE']]
    pdg_mva_one = eval_proba[p_types != pdg_code_one, 1]
    pdg_dll_one = (eval_data[comb_dll])[p_types != pdg_code_one].values

    pdg_code_two = particle_pdg_codes[params['PARTICLE']]
    pdg_mva_two = eval_proba[p_types == pdg_code_two, 1]
    pdg_dll_two = (eval_data[comb_dll])[p_types == pdg_code_two].values

    effs_p_one, effs_p_two, eff_errs_p_one, eff_errs_p_two = \
    get_miss_and_eff(pdg_mva_one, pdg_mva_two, n_bins, 0, 1)

    effs_p_one2, effs_p_two2, eff_errs_p_one2, eff_errs_p_two2 = \
    get_miss_and_eff(pdg_dll_one, pdg_dll_two, n_bins, dll_min[params['PARTICLE']], dll_max[params['PARTICLE']])

    #plt.errorbar(effs_p_two, effs_p_one, xerr=eff_errs_p_two, yerr=eff_errs_p_one, fmt='none', ecolor=color)
    plt.plot(effs_p_two, effs_p_one, c='b', label='MVA Out', linewidth=2)
    plt.plot(effs_p_two2, effs_p_one2, c='r', label=comb_dll, linewidth=2)

    plt.ylabel("Overall MissID Efficiency / %", size=15)
    plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " Efficiency / %", size=15)
    plt.legend(loc='best')

    plt.xticks(numpy.arange(0, 101, 10))
    plt.xlim(0, 100)

    plt.grid(True, lw = 2, ls = ':', c = '.75')

    if log:
        plt.yscale('log', nonposy='clip')

    if log:
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                    "-IDEff_V_" + "OverallMissIDEff_Logy.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                    "-IDEff_V_" + "OverallMissIDEff_Logy.png")
    else:
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                    "-IDEff_V_" + "OverallMissIDEff_Liny.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                    "-IDEff_V_" + "OverallMissIDEff_Liny.png")


    plt.close()

    return 1

def MVADLL(params, eval_data, eval_proba, eval_labels, features, log=False, signal=1, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    p_types = numpy.abs(eval_data.MCParticleType.values)
    comb_dll = comb_dlls[params['PARTICLE']]

    plt.figure(figsize=(10,7))

    dll_data2 = eval_data[comb_dll][eval_labels==signal].values
    dll_data = dll_data2[(dll_data2 >= dll_min[params['PARTICLE']]) * (dll_data2 < dll_max[params['PARTICLE']])]
    mva_data = eval_proba[eval_labels==signal, 1]
    mva_data = mva_data[(dll_data2 >= dll_min[params['PARTICLE']]) * (dll_data2 < dll_max[params['PARTICLE']])]

    if log:
        plt.hist2d(dll_data, mva_data, bins=n_bins_2d, norm = mpl.colors.LogNorm())
    else:
        plt.hist2d(dll_data, mva_data, bins=n_bins_2d)

    plt.colorbar()

    plt.ylim(0, 1)
    plt.xlim(dll_min[params['PARTICLE']], dll_max[params['PARTICLE']])

    plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " " + comb_dll)
    plt.ylabel(params['TRACK'] + " " + params['PARTICLE'] + " MVA Output")

    if log and signal:
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                    "-MVAOutVDLL-Signal_Logz.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                    "-MVAOutVDLL-Signal_Logz.png")
    elif log and not signal:
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                    "-MVAOutVDLL-Background_Logz.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                    "-MVAOutVDLL-Background_Logz.png")
    elif not log and signal:
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                    "-MVAOutVDLL-Signal_Linz.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                    "-MVAOutVDLL-Signal_Linz.png")
    else:
        plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                    "-MVAOutVDLL-Background_Linz.pdf")
        plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                    "-MVAOutVDLL-Background_Linz.png")
    plt.close()


    return 1

# Pur V MVA
def PurityVMVAOut(params, eval_data, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    plt.figure(figsize=(10,7))

    purs, pur_errs, edges, mva_errs = get_pur(eval_proba[:, 1], eval_labels, 100, 0, 1)

    plt.scatter(edges, purs)
    plt.errorbar(edges, purs, xerr=mva_errs, yerr=pur_errs, fmt='none')

    plt.ylim(0,100)
    plt.xlim(0, 1)
    plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " MVA Output", size=15)
    plt.ylabel('Purity / %', size=15)

    plt.xticks(numpy.arange(0, 1.01, .10))
    plt.yticks(numpy.arange(0, 101, 10))

    plt.grid(True, lw = 2, ls = ':', c = '.75')

    plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                "-PurityVMVAOut.pdf")
    plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                "-PurityVMVAOut.png")

    plt.close()

    return 1

# Pur V MVA
def PurityVCombDLL(params, eval_data, eval_proba, eval_labels, features, path="pic"):

    import os

    if not os.path.exists(path + "/pdf"):
        os.makedirs(path + "/pdf")

    if not os.path.exists(path + "/png"):
        os.makedirs(path + "/png")

    plt.figure(figsize=(10,7))

    comb_dll = comb_dlls[params['PARTICLE']]

    purs, pur_errs, edges, mva_errs = get_pur(eval_data[comb_dll].values, eval_labels, 100,
                                          dll_min[params['PARTICLE']], dll_max[params['PARTICLE']])

    plt.scatter(edges, purs)
    plt.errorbar(edges, purs, xerr=mva_errs, yerr=pur_errs, fmt='none')

    plt.ylim(0,100)
    plt.xlim(dll_min[params['PARTICLE']], dll_max[params['PARTICLE']])
    plt.xlabel(params['TRACK'] + " " + params['PARTICLE'] + " " + comb_dll, size=15)
    plt.ylabel('Purity / %', size=15)

    plt.yticks(numpy.arange(0, 101, 10))

    plt.grid(True, lw = 2, ls = ':', c = '.75')

    plt.savefig(path + "/pdf" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                "-PurityVCombDLL.pdf")
    plt.savefig(path + "/png" + "/" + params['TRACK'] + "_" + params['PARTICLE'] + " " + \
                "-PurityVCombDLL.png")

    plt.close()

    return 1

# MAIN function

def all_figures(params, eval_data, eval_proba, eval_labels, features, path="pic"):

    Inputs(params, eval_data, eval_proba, eval_labels, features, log=False, path=path)
    Inputs(params, eval_data, eval_proba, eval_labels, features, log=True, path=path)

    MVAVInputs(params, eval_data, eval_proba, eval_labels, features, path=path)
    DLLVInputs(params, eval_data, eval_proba, eval_labels, features, path=path)

    MVAEffForDLLCut(params, eval_data, eval_proba, eval_labels, features, path=path)
    DLLEffForMVACut(params, eval_data, eval_proba, eval_labels, features, path=path)

    DLLEffVTrackP(params, eval_data, eval_proba, eval_labels, features, path=path)
    DLLEffVTrackPt(params, eval_data, eval_proba, eval_labels, features, path=path)

    MVAEffVTrackP(params, eval_data, eval_proba, eval_labels, features, path=path)
    MVAEffVTrackPt(params, eval_data, eval_proba, eval_labels, features, path=path)

    CombDLL(params, eval_data, eval_proba, eval_labels, features, log=False, path=path)
    CombDLL(params, eval_data, eval_proba, eval_labels, features, log=True, path=path)

    MVAOut(params, eval_data, eval_proba, eval_labels, features, log=False, path=path)
    MVAOut(params, eval_data, eval_proba, eval_labels, features, log=True, path=path)

    EffPurity(params, eval_data, eval_proba, eval_labels, features, log=False, path=path)

    EffMissIDEff(params, eval_data, eval_proba, eval_labels, features, path=path)

    EffOverallMissIDEff(params, eval_data, eval_proba, eval_labels, features, log=False, path=path)
    EffOverallMissIDEff(params, eval_data, eval_proba, eval_labels, features, log=True, path=path)

    MVADLL(params, eval_data, eval_proba, eval_labels, features, log=False, signal=1, path=path)
    MVADLL(params, eval_data, eval_proba, eval_labels, features, log=False, signal=0, path=path)
    MVADLL(params, eval_data, eval_proba, eval_labels, features, log=True, signal=1, path=path)
    MVADLL(params, eval_data, eval_proba, eval_labels, features, log=True, signal=0, path=path)

    PurityVMVAOut(params, eval_data, eval_proba, eval_labels, features, path=path)
    PurityVCombDLL(params, eval_data, eval_proba, eval_labels, features, path=path)

    return 1


