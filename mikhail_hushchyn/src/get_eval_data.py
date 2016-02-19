__author__ = 'mikhail91'

import os
import numpy
import pandas
import root_numpy

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
def get_eval_data(params, location='http'):

    ###################################################
    # Check train directory
    ###################################################

    # Results directory
    WORKPATH = params['TRAINDIR']
    # LOG file
    LOGFILE = WORKPATH + "/" + params['TRACK'] + "-" + params['PARTICLE'] + "-eval.log"
    # viewed_files file
    VIEWEDFILES = WORKPATH + "/viewed_files.txt"

    if not os.path.exists(WORKPATH):
        os.makedirs(WORKPATH)
        LOG = open(LOGFILE, 'w')

        VIEWED = open(VIEWEDFILES, 'w')
        VIEWED.write("")
        VIEWED.close()

        VIEWED = list(numpy.loadtxt(VIEWEDFILES, dtype='S', delimiter='\n', comments='#'))
        LOG.write('Hello!\n')
        LOG.flush()
        print "Folder with results was created:" + WORKPATH
    else:
        LOG = open(LOGFILE, 'a')
        VIEWED = list(numpy.loadtxt(VIEWEDFILES, dtype='S', delimiter='\n', comments='#', ndmin=1))
        print "The directory exist."



    ###################################################
    # Initialise
    ###################################################

    # Open network config file
    net_config = numpy.loadtxt(params['NETCONFIG'], dtype='S', delimiter='\n', comments='!')

    # Open eval config file
    # eval_params = "None"
    # if params['EVALPARAMSFILE'] != "None":
    #     eval_params = numpy.loadtxt(params['EVALPARAMSFILE'], dtype='S', delimiter='\n', comments='#')

    # Eval data files
    eval_files = numpy.loadtxt(params['EVALFILES'], dtype='S', delimiter='\n', comments='#')



    # Open training config file
    train_config = numpy.loadtxt(params['TRAINCONFIG'], dtype='S', delimiter='\n', comments='#')

    # Training data files
    train_files = numpy.loadtxt(params['TRAINFILES'], dtype='S', delimiter='\n', comments='#')

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
    mc_track_sel_eval_name = train_config[4]
    mc_track_sel_eval_file = params['TRAINCONFIGDIR'] + train_config[4]

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

    features = numpy.array(features)
    spectator_features = numpy.array(spectator_features)

    # Tracks type selection ???
    track_type_sel = track_selections[track_type]

    # Tracks preselection
    track_sel = numpy.loadtxt(track_sel_file, dtype='S', delimiter='\n', comments='#', ndmin=1)

    # MC tracks eval selectiom
    mc_track_sel_eval = numpy.loadtxt(mc_track_sel_eval_file, dtype='S', delimiter='\n', comments='#', ndmin=1)

    # Reweighting selection
    if reweight_opt == "ReweightRICH2" :
        reweight_sel = createRICH2HitsReweightSel()
    else:
        reweight_sel = ""

    # Number of training tracks
    n_training_tracks = train_config[7]

    # Number of eval tracks
    n_eval_tracks = int(train_config[8])//10 # !!!!!!!!!!!!!!!!!

    # NUmber of features
    n_var = len(features)

    # Ghost accept fraction
    ghost_accept_frac = get_params(params, 'GHOSTACPTFRAC', 1.0)

    if ghost_accept_frac < 1.0 and "EqualMix" == training_mix:
        print "Ghost Fraction < 1 makes no sense with equal mixture training"


    # Sanity checks
    if particle_type == bkg_type:
        print "Background and PID types the same " + particle_type + " !!!"
        LOG.write("Background and PID types the same " + particle_type + " !!!\n")
        LOG.flush()

    if "Ghost" == particle_type and\
      ("NoGhosts" == ghost_treatment_training or\
       "BTracksOnly.txt" == mc_track_sel_training_name or\
        ghost_accept_frac <= 0.0):

        print "Cannot train ghost ID network whilst rejecting ghosts !!!"
        LOG.write("Cannot train ghost ID network whilst rejecting ghosts !!!\n")
        LOG.flush()

    LOG.write("Initialisation is completed.\n")
    LOG.flush()
    print "Initialisation is completed."


    #############################################
    # Read in training data
    #############################################

    # Evaluation data files
    eval_set = get_params(params, 'EVALDATA', "Mixture")

    # Count tracks overall
    all_tracks = 0
    selected_tracks = 0


    # Overall selection
    overall_eval_sel = get_overal_track_sel(track_type_sel, track_sel, mc_track_sel_eval)

    # Full selection for file determination
    if reweight_sel != "":
        eval_data_file_sel = overall_eval_sel + " && " + " ( " +  reweight_sel + " ) "
    else:
        eval_data_file_sel = overall_eval_sel

    # Ghost selection
    ghost_accept_sel = create_ghost_accept_sel(ghost_accept_frac)

    # Combined train selection
    if ghost_accept_sel != "":
        combined_eval_sel = eval_data_file_sel + " && (" + ghost_accept_sel + " )"
    else :
        combined_eval_sel = eval_data_file_sel



    # Data Frames for eval
    try:
        data_eval = pandas.read_csv(WORKPATH + '/data_eval.csv', usecols=['MCParticleType'])

        selected_tracks = len(data_eval)

        data_eval = pandas.DataFrame()

    except:
        data_eval = pandas.DataFrame()


    # Loop over eval file list as far as required
    LOG.write("Start reading files.\n")
    LOG.flush()
    print "Start reading files.\n"

    n_files_used = 0

    for data_file in eval_files:

        # Found enough tracks ?
        if selected_tracks >= int(n_eval_tracks):
            break

        n_files_used += 1

        # Get data file path and tree name
        if location == 'local':
            data_file_path, data_file_tree = data_file.split(':')
        elif location == 'http':
            data_file_path_raw, data_file_tree = data_file.split(':')
            data_file_path = data_file_path_raw.replace("/r02/lhcb/jonesc/ANNPID/", "http://www.hep.phy.cam.ac.uk/~jonesc/lhcb/PID/")
            data_file_path = data_file_path.replace("/r03/lhcb/jonesc/ANNPID/", "http://www.hep.phy.cam.ac.uk/~jonesc/lhcb/PID/")
        else:
            print "File path is incorrect. Check the configs."
            return 0

        # A file was viewed before?
        if data_file_path in VIEWED:
            continue

        LOG.write("File " + data_file_path + " is readed.\n")
        LOG.flush()
        print "File " + data_file_path + " is readed.\n"

        try:
            # Open data file and convert it to csv
            branches = root_numpy.list_branches(data_file_path, treename=data_file_tree)
            branches = numpy.array(branches)

            eval_tree = root_numpy.root2array(data_file_path,
                                                  treename=data_file_tree,
                                                  branches=branches[branches != 'piplus_OWNPV_COV_'],
                                                  selection = combined_eval_sel)[::10] # !!!!!!!!!!!!!!!!!!!!!!!!!!!

            eval_tree = pandas.DataFrame(data=eval_tree, columns=branches[branches != 'piplus_OWNPV_COV_'])
        except:
            LOG.write("Cannot read file: " + data_file_path + "\n")
            LOG.flush()
            print "Cannot read file: " + data_file_path
            continue

        # Count selected tracks for this file
        selected_tracks_file = 0

        rows_eval = []

        # Loop over entry list
        for irow in range(0, len(eval_tree)):

            data_row = eval_tree.irow([irow])

            mc_particle_type = numpy.abs(data_row.MCParticleType.values[0])

            # Count all tracks considered
            all_tracks += 1

            # Ghost treatment
            if 0 == mc_particle_type and not keep_ghost_eval:
                continue


            rows_eval.append(irow)


            # count tracks
            selected_tracks += 1
            selected_tracks_file += 1;

            # Found enough tracks ?
            if selected_tracks >= int(n_eval_tracks):
                break

        data_eval = eval_tree.irow(rows_eval)

        VIEWED.append(data_file_path)
        numpy.array(VIEWED).tofile(VIEWEDFILES, sep="\n")
        LOG.write("Link to file was writen in viewed_files.txt\n")
        LOG.flush()

        if os.path.exists(WORKPATH + '/data_eval.csv'):
            data_eval.to_csv(WORKPATH + '/data_eval.csv', mode='a', header=False)
        else:
            data_eval.to_csv(WORKPATH + '/data_eval.csv', mode='a', header=True)

        LOG.write("n_eval_tracks = " + str(n_eval_tracks) + "\n")
        LOG.write("selected_tracks = " + str(selected_tracks) + "\n")
        LOG.write("selected_tracks_file = " + str(selected_tracks_file) + "\n")
        LOG.flush()

        LOG.write("Data is written.\n")
        LOG.write("###########################################################\n")
        LOG.flush()
        print "Data is written."

        # Found enough tracks ?
        if selected_tracks >= int(n_eval_tracks):
            break


    LOG.write("Was selected " + str(selected_tracks) + " tracks.\n")
    LOG.write("Was used " + str(n_files_used) + " files.\n")
    LOG.write("Reading train data is completed.\n")
    LOG.flush()
    print "Reading train data is completed.\n"


    LOG.write("Writing train data is completed.\n")
    LOG.flush()
    print "Writing train data is completed.\n"


    LOG.close()

    return WORKPATH