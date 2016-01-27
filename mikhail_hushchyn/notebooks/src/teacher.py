__author__ = 'mikhail91'

import os
import numpy
import pandas
import root_numpy
from rep.estimators import TMVAClassifier

all_pid_type = 999999
particle_pdg_codes = {"all": all_pid_type,
                    "ghost": 0,
                    "electron": 11,
                    "muon": 13,
                    "pion": 211,
                    "kaon": 321,
                    "proton": 2212}

comb_dlls = {"electron": "CombDLLe",
             "muon": "CombDLLmu",
             "pion": "CombDLLpi",
             "kaon": "CombDLLk",
             "proton": "CombDLLp",
             "ghost": "TrackGhostProbability"}

track_selections = {"Long": "abs(TrackType-3) < 0.1",
                    "Upstream": "abs(TrackType-4) < 0.1",
                     "Downstream": "abs(TrackType-5) < 0.1"}

GeV = 1000
limits = {"TrackP": [100*GeV, 0],
          "TrackPt": [10*GeV, 0],
          "CombDLLe": [20, -20],
          "CombDLLmu": [20, -20],
          "CombDLLpi": [150, -150],
          "CombDLLk": [150, -150],
          "CombDLLp": [150, -150] }

mass_hypos = [11,13,211,321,2212,0]


# Secondary functions
def createRICH2HitsReweightSel():

    selS = "( 0.00108528 +" + \
           "( -1.0822e-05  * NumRich2Hits        ) +" + \
           "( 3.51537e-08  * pow(NumRich2Hits,2) ) +" + \
           "( -4.6664e-11  * pow(NumRich2Hits,3) ) +" + \
           "( 3.16421e-14  * pow(NumRich2Hits,4) ) +" + \
           "( -8.69538e-18 * pow(NumRich2Hits,5) ) +" + \
           "( 1.14595e-21  * pow(NumRich2Hits,6) ) )" + \
           " > rndm"

    return selS

def get_params(params, key_val, default):

    if params.has_key(key_val):
        param_val = params[key_val]
    else:
        param_val = default

    return param_val


def get_overal_track_sel(track_type_sel, track_sel, mc_track_sel_training):

    overall_training_sel = ""

    # add track_type_sel
    for sel in [track_type_sel]:

        if overall_training_sel == "":
            overall_training_sel += " ( " + sel + " ) "

        else:
            overall_training_sel += " && " + " ( " + sel + " ) "

    # add track_sel
    for sel in track_sel:

        if overall_training_sel == "":
            overall_training_sel += " ( " + sel + " ) "

        else:
            overall_training_sel += " && " + " ( " + sel + " ) "

    # add mc_track_sel_training
    for sel in mc_track_sel_training:

        if overall_training_sel == "":
            overall_training_sel += " ( " + sel + " ) "

        else:
            overall_training_sel += " && " + " ( " + sel + " ) "

    return overall_training_sel


def create_ghost_accept_sel(ghost_accept_frac):
    sel = ""
    if ghost_accept_frac < 1.0 :
        sel = "( MCParticleType != 0 || rndm < " + str(ghost_accept_frac) + " )"
    return sel


# Main code!
def teacher(params, location='http'):

    ###################################################
    # Check train directory
    ###################################################

    # Results directory
    WORKPATH = params['TRAINDIR'] + "/" + params['PARTICLE'] + "/" + params['TRACK']
    # LOG file
    LOGFILE = WORKPATH + "/" + params['TRACK'] + "-" + params['PARTICLE'] + ".log"

    if not os.path.exists(WORKPATH):
        os.makedirs(WORKPATH)
        LOG = open(LOGFILE, 'w')
        LOG.write('Hello!\n')
        print "Folder with results was created:" + WORKPATH
    else:
        print "Classifier with these parameters was trained. Remove TRAINDIR directory."
        return 0



    ###################################################
    # Initialise
    ###################################################

    # Running options
    do_train = params['DOTRAIN']
    do_eval = params['DOEVAL']

    # Open network config file
    net_config = numpy.loadtxt(params['NETCONFIG'], dtype='S', delimiter='\n', comments='!')

    # Open training config file
    train_config = numpy.loadtxt(params['TRAINCONFIG'], dtype='S', delimiter='\n', comments='#')

    # Open MVA specific config file
    mva_config = numpy.loadtxt(params['MVACONFIG'], dtype='S', delimiter='\n', comments='#')

    # Open eval config file
    eval_params = "None"
    if params['EVALPARAMSFILE'] != "None":
        eval_params = numpy.loadtxt(params['EVALPARAMSFILE'], dtype='S', delimiter='\n', comments='#')

    # Training data files
    train_files = numpy.loadtxt(params['TRAINFILES'], dtype='S', delimiter='\n', comments='#')

    # Eval data files
    eval_files = numpy.loadtxt(params['EVALFILES'], dtype='S', delimiter='\n', comments='#')

    # Read the particle type
    particle_type = net_config[0]
    particle_pdg = particle_pdg_codes[particle_type]
    particle_comb_dll = comb_dlls[particle_type]

    # Read the track type
    track_type = net_config[1]
    track_type_sel = track_selections[track_type]

    # Track selection file
    track_sel_file = params['TRACKSELCONFIGDIR'] + net_config[2]

    # Background type
    bkg_type = train_config[0]
    bkg_pdg = particle_pdg_codes[bkg_type]

    # Ghost treatment for training
    ghost_treatment_training = train_config[1]
    keep_ghost_training = train_config[1] != "NoGhosts"

    # Ghost treatment for evaluation
    ghost_treatment_eval = train_config[2]
    keep_ghost_eval = train_config[2] != "NoGhosts"

    # MC track training selection
    mc_track_sel_training_name = train_config[3]
    mc_track_sel_training_file = params['TRAINCONFIGDIR'] + train_config[3]

    # MC track eval selection
    mc_track_sel_eval_file = train_config[4]

    # Signal/Background mix
    training_mix = train_config[5]

    # Reweighting
    reweight_opt = train_config[6]

    # Read the network type
    mva_type = net_config[3]

    # Read network parameters file name
    param_file_name = net_config[4]


    # Read train and spectator features
    features = []
    spectator_features = []

    for var in net_config[5:]:
        if var.find('#') == -1:
            features.append(var)
        else:
            spectator_features.append(var[1:])

    # Tracks type selection ???
    track_type_sel = track_selections[track_type]

    # Tracks preselection
    track_sel = numpy.loadtxt(track_sel_file, dtype='S', delimiter='\n', comments='#')

    # MC tracks training selectiom
    mc_track_sel_training = numpy.loadtxt(mc_track_sel_training_file, dtype='S', delimiter='\n', comments='#')

    # Reweighting selection
    if reweight_opt == "ReweightRICH2" :
        reweight_sel = createRICH2HitsReweightSel()
    else:
        reweight_sel = ""

    # Number of training tracks
    n_training_tracks = train_config[7]

    # Number of eval tracks
    n_eval_tracks = train_config[8]

    LOG.write("Initialisation is completed.\n")
    print "Initialisation is completed.\n"

    ###############################################
    # Configure the teacher
    ###############################################

    # Number of inputs and hidden nodes
    if params.has_key('MLPHIDDENLAYERSCALEF'):
        layer_two_scale = params['MLPHIDDENLAYERSCALEF']
    else:
        layer_two_scale = 1.2

    n_var = len(features)

    n_hidden_nodes = int(n_var * layer_two_scale)


    # Ghost accept fraction
    ghost_accept_frac = get_params(params, 'GHOSTACPTFRAC', 1.0)

    if ghost_accept_frac < 1.0 and "EqualMix" == training_mix:
        print "Ghost Fraction < 1 makes no sense with equal mixture training"


    # Evaluation data files
    eval_set = get_params(params, 'EVALDATA', "Mixture")


    # Reuse files for eval
    reuse_files = get_params(params, 'REUSETRAININGFILES', 0)


    # TMVA parameters
    # General
    tmva_method = get_params(params, "TMVAMETHOD","MLP")
    tmva_var_transform = get_params(params,"TMVAVARTRANSFORM","None")
    tmva_validation_frac = get_params(params, "TMVAVALIDATIONFRAC",0.3)
    tmva_test_frac = get_params(params, "TMVATESTFRAC",0.3)
    tmva_use_regulator = get_params(params, "TMVAUSEREGULATOR","false")
    # MLP specific
    tmva_mlp_neuron_type = get_params(params, "TMVAMLPNEURONTYPE","sigmoid")
    tmva_mlp_method = get_params(params, "TMVAMLPMETHOD","BP")
    tmva_mlp_n_cycles = get_params(params, "TMVAMLPNCYCLES","500")
    tmva_mlp_estimator_type = get_params(params, "TMVAMLPESTIMATORTYPE","CE")
    tmva_mlp_conv_improve = get_params(params, "TMVAMLPCONVIMPROVE","1e-16")
    tmva_mlp_conv_test = get_params(params, "TMVAMLPCONVTEST",-1)
    # BDT specific
    tmva_bdt_boost_type = get_params(params, "TMVABDTBOOSTTYPE","AdaBoost")
    tmva_bdt_n_trees = get_params(params, "TMVABDTNTREES","800")
    tmva_bdt_prune_method = get_params(params, "TMVABDTPRUNEMETHOD","NoPruning")
    tmva_bdt_max_tree_depth = get_params(params, "TMVABDTMAXTREEDEPTH","3")


    # Sanity checks
    if particle_type == bkg_type:
        print "Background and PID types the same " + particleType + " !!!"
        LOG.write("Background and PID types the same " + particleType + " !!!\n")

    if "Ghost" == particle_type and\
      ("NoGhosts" == ghost_treatment_training or\
       "NoGhosts" == ghost_treatment_eval or\
       "BTracksOnly.txt" == mc_track_sel_training_name or\
        ghost_accept_frac <= 0.0):

        print "Cannot train ghost ID network whilst rejecting ghosts !!!"
        LOG.write("Cannot train ghost ID network whilst rejecting ghosts !!!\n")

    print "Configuration is completed."
    LOG.write("Configuration is completed.\n")


    #############################################
    # Read in training data
    #############################################

    # Count tracks overall
    all_tracks = 0
    selected_tracks = 0
    signal_tracks = 0
    background_tracks = 0
    test_tracks = 0

    selected_tracks_by_type = {11: 0, 13: 0, 211: 0, 321: 0, 2212: 0, 0: 0}


    # Overall selection
    overall_training_sel = get_overal_track_sel(track_type_sel, track_sel, mc_track_sel_training)

    # Full selection for file determination
    if reweight_sel != "":
        train_data_file_sel = overall_training_sel + " && " + " ( " +  reweight_sel + " ) "
    else:
        train_data_file_sel = overall_training_sel

    # Ghost selection
    ghost_accept_sel = create_ghost_accept_sel(ghost_accept_frac)

    # Combined train selection
    if ghost_accept_sel != "":
        combined_train_sel = train_data_file_sel + " && (" + ghost_accept_sel + " )"
    else :
        combined_train_sel = train_data_file_sel

    # Do we want equal amounts of all particle types ?
    equal_by_type = ( "EqualMix" == training_mix )
    if ghost_accept_frac > 0.0 and keep_ghost_training:
        n_hypos = 1. * len(mass_hypos)
    else:
        n_hypos = 1. * len(mass_hypos) - 1.
    n_per_type = 1 + ( float(n_training_tracks) / n_hypos )


    # Data Frames for trainer
    data_train_signal = pandas.DataFrame()
    data_train_bkg = pandas.DataFrame()
    data_test_signal = pandas.DataFrame()
    data_test_bkg = pandas.DataFrame()


    # Loop over training file list as far as required
    LOG.write("Start reading files.\n")
    print "Start reading files.\n"

    n_files_used = 0

    for data_file in train_files:

        LOG.write("File " + data_file + " is readed.\n")
        print "File " + data_file + " is readed.\n"

        n_files_used += 1

        # Get data file path and tree name
        if location == 'local':
            data_file_path, data_file_tree = data_file.split(':')
        elif location == 'http':
            data_file_path, data_file_tree = data_file.split(':')
            data_file_path = data_file_path.replace("/r02/lhcb/jonesc/ANNPID/", "http://www.hep.phy.cam.ac.uk/~jonesc/lhcb/PID/")
        else:
            print "File path is incorrect. Check the configs."
            return 0

        # Open data file and convert it to csv
        branches = root_numpy.list_branches(data_file_path, treename=data_file_tree)
        branches = numpy.array(branches)

        training_tree = root_numpy.root2array(data_file_path,
                                              treename=data_file_tree,
                                              branches=branches[branches != 'piplus_OWNPV_COV_'],
                                              selection = combined_train_sel)

        training_tree = pandas.DataFrame(data=training_tree, columns=branches[branches != 'piplus_OWNPV_COV_'])

        # MC type
        mc_particle_type = training_tree.MCParticleType.values

        # Count selected tracks for this file
        selected_tracks_file = 0
        test_tracks_file = 0

        # Loop over entry list
        for irow in range(0, len(training_tree)):

            data_row = training_tree.irow([irow])

            mc_particle_type = numpy.abs(data_row.MCParticleType.values[0])

            # Count all tracks considered
            all_tracks += 1

            # Ghost treatment
            if 0 == mc_particle_type and not keep_ghost_training:
                continue

            # True or fake target
            target = mc_particle_type == particle_pdg

            # Equal mix ?
            if (equal_by_type and selected_tracks_by_type[mc_particle_type] >= n_per_type) or \
                not selected_tracks_by_type.has_key(mc_particle_type) :
                continue

            # Background type selection for training
            if not ( target or all_pid_type == bkg_pdg or mc_particle_type == int(bkg_pdg) ):
                continue

            # Count signal and background
            if target:
                signal_tracks += 1
            else:
                background_tracks += 1

            # Fill an input array for the teacher ?????

            # Make sure min and max are filled for spectators ?????

            use_for_testing = "False"

            # Set inputs and target output
            use_for_testing = numpy.random.rand() < float(tmva_test_frac)

            if target:

                if not use_for_testing:
                    data_train_signal = pandas.concat([data_train_signal, data_row], ignore_index=True) # data_row[features]
                else:
                    data_test_signal = pandas.concat([data_test_signal, data_row], ignore_index=True) # data_row[features]

            else:

                if not use_for_testing:
                    data_train_bkg = pandas.concat([data_train_bkg, data_row], ignore_index=True) # data_row[features]
                else:
                    data_test_bkg = pandas.concat([data_test_bkg, data_row], ignore_index=True) # data_row[features]


            # count tracks
            if not use_for_testing:
                selected_tracks += 1
                selected_tracks_file += 1;
                selected_tracks_by_type[mc_particle_type] += 1
            else:
                test_tracks += 1
                test_tracks_file += 1

            # Found enough tracks ?
            if selected_tracks >= int(n_training_tracks):
                break



        # Found enough tracks ?
        if selected_tracks >= n_training_tracks:
            break

        # Used 1/2 of the training files ?
        if n_files_used > len(train_files)/2:
            break

    LOG.write("Was selected " + str(selected_tracks) + " tracks.\n")
    LOG.write("Was used " + str(n_files_used) + " files.\n")
    LOG.write("Selected tracks by type: " + str(selected_tracks_by_type) + ".\n")
    LOG.write("Reading train data is completed.\n")
    print "Reading train data is completed.\n"

    data_train_signal.to_csv(params['TRAINDIR'] + '/data_train_signal.csv')
    data_train_bkg.to_csv(params['TRAINDIR'] + '/data_train_bkg.csv')
    data_test_signal.to_csv(params['TRAINDIR'] + '/data_test_signal.csv')
    data_test_bkg.to_csv(params['TRAINDIR'] + '/data_test_bkg.csv')

    LOG.write("Writing train data is completed.\n")
    print "Writing train data is completed.\n"


    ##############################################
    # For TMVA, must setup the method here
    ##############################################

    if params['MVATYPE'] == 'TMVA':

        # Prepare the data
        config = "V:nTrain_Signal=0:nTrain_Background=0:SplitMode=Random"
        if "EqualSigBck" == training_mix:
            config += ":NormMode=EqualNumEvents"
        else:
            config += ":NormMode=None"
        # tmvaFactory->PrepareTrainingAndTestTree( cuts, config.c_str() ) !!!!!!!!


        # TMVA Method name
        name = particle_type + "_" + track_type + "_TMVA"

        # Sort of TMVA classifier
        if "MLP" == tmva_method:

            # Construct MLP options
            tmva = TMVAClassifier(method='kMLP',
                                  features=features,
                                  factory_options="V:!Silent:!Color:!DrawProgressBar",
                                  H='true',
                                  V='true',
                                  EpochMonitoring='true',
                                  HiddenLayers=str(n_hidden_nodes),
                                  UseRegulator=tmva_use_regulator)

            if int(tmva_mlp_conv_test) > 0:
                tmva.set_params(ConvergenceImprove = tmva_mlp_conv_improve)
                tmva.set_params(ConvergenceTests = tmva_mlp_conv_test)
            if "DEFAULT" != tmva_var_transform:
                tmva.set_params(VarTransform = tmva_var_transform)
            if "DEFAULT" != tmva_mlp_n_cycles:
                tmva.set_params(NCycles = tmva_mlp_n_cycles)
            if "DEFAULT" != tmva_mlp_neuron_type:
                tmva.set_params(NeuronType = tmva_mlp_neuron_type)
            if "DEFAULT" != tmva_mlp_method:
                tmva.set_params(TrainingMethod = tmva_mlp_method)
            if "DEFAULT" != tmva_mlp_estimator_type:
                tmva.set_params(EstimatorType = tmva_mlp_estimator_type)
            # tmvaFactory->BookMethod( TMVA::Types::kMLP, name.str().c_str(), opts.str().c_str() );

        elif "BDT" == tmva_method:

            # BDT opts
            tmva = TMVAClassifier(method='kBDT',
                                  features=features,
                                  factory_options="V:!Silent:!Color:!DrawProgressBar",
                                  H='false',
                                  V='true',
                                  NTrees=tmva_bdt_n_trees)

            # opts += ":UseRegulator=" + tmva_use_regulator
            if "DEFAULT" != tmva_var_transform:
                tmva.set_params(VarTransform = tmva_var_transform)
            if "DEFAULT" != tmva_bdt_boost_type:
                tmva.set_params(BoostType = tmva_bdt_boost_type)
            if "DEFAULT" != tmva_bdt_prune_method:
                tmva.set_params(PruneMethod = tmva_bdt_prune_method)
            if "CostComplexity" == tmva_bdt_prune_method or "ExpectedError" == tmva_bdt_prune_method:
                tmva.set_params(PruneStrength = -1)
            if "DEFAULT" != tmva_bdt_max_tree_depth:
                tmva.set_params(MaxDepth = tmva_bdt_max_tree_depth)
            if float(tmva_validation_frac) > 0 and "NoPruning" != tmva_bdt_prune_method:
                tmva.set_params(PruningValFraction = tmva_validation_frac)
            # tmvaFactory->BookMethod( TMVA::Types::kBDT, name.str().c_str(), opts.str().c_str() );


    ###############################################
    # Train the network
    ###############################################

    if do_train == "Yes":
        train_data = pandas.concat([data_train_signal, data_train_bkg], axis=0)
        train_labels = numpy.concatenate((numpy.ones(len(data_train_signal)), numpy.zeros(len(data_train_bkg))), axis=0)

        if "EqualSigBck" == training_mix:
            k = 1. * len(data_train_signal)/len(data_train_bkg)
        else:
            k = 1.

        sample_weight = numpy.concatenate((numpy.ones(len(data_train_signal)),
                                            k * numpy.ones(len(data_train_bkg))), axis=0)

        tmva.fit(train_data, train_labels, sample_weight=sample_weight)

        LOG.write('Training complete.\n')
        print('Training complete')
        LOG.close()

    if do_eval == "Yes":
        pass

    return tmva, data_train_signal, data_train_bkg, data_test_signal, data_test_bkg