__author__ = 'mikhail91'
import numpy, pandas, root_numpy

def get_number_particles(files_http, particles_pdg, selection, log_file_name='get_number_particles.log'):
    """
    Returns number of particles of each type in each data file.
    :param files_http: list of strings, list of http for the each data file.
    :param particles_pdg: list of integers, list of pdg codes of the particles.
    :param selection: string, selection criteria for the particles.
    :param log_file_name: string, name of the log file.
    :return: pandas.DataFrame
    """

    numbers_particles = numpy.zeros((len(files_http), len(particles_pdg)))

    LOG = open(log_file_name, 'w')
    LOG.write('Particles pdgs: ' + str(particles_pdg) + '\n')
    LOG.write('Selection: ' + selection + '\n')
    LOG.flush()

    for num, (one_file_https, one_file_tree) in enumerate(files_http):

        success = 0
        while success != 1:
            try:
                mc_particle_type = root_numpy.root2array(filenames=one_file_https,
                                                         treename=one_file_tree,
                                                         branches='MCParticleType',
                                                         selection=selection)

                for num_pdg, one_pdg in enumerate(particles_pdg):

                    numbers_particles[num, num_pdg] = (numpy.abs(mc_particle_type) == one_pdg).sum()

                LOG.write(str(num) + '. ' + one_file_https + '\n')
                LOG.write('Numbers of particles: ' + str(numbers_particles[num, :]) + '\n')
                LOG.flush()
                success = 1

            except:
                LOG.write(one_file_https + ' is not readed.' + '\n')
                LOG.flush()

    numbers_particles_df = pandas.DataFrame(data=numbers_particles, columns=[str(i) for i in particles_pdg])
    numbers_particles_df['http'] = numpy.array(files_http)[:, 0]
    numbers_particles_df['tree_name'] = numpy.array(files_http)[:, 1]

    return numbers_particles_df

from sklearn.cross_validation import train_test_split
import gc, os
import sys

def generate_data_sample(numbers_particles, n_tracks, selection, file_path, log_path, readed_files_txt):
    """
    Generates data sample from different decays.
    :param numbers_particles: pandas.DataFrame, number of particles of each type in each data file.
    :param n_tracks: int, number of tracks of each particle type.
    :param selection: string, selection criteria for the particles.
    :param file_path: string, name of the data sample file.
    :param log_path: string, name of the log file.
    :param readed_files_txt: string, name of the file which contains https of all read files.
    Files from this file will not be read. This is needed for the warm start.
    :return: 1
    """

    # Estimate how many track of the each particle from the each file should be taken
    particles = numbers_particles.columns.drop(['http', 'tree_name'])
    part = 1. * n_tracks / numbers_particles[particles].sum()


    # Try to create or open LOG file
    if not os.path.exists(log_path):

        LOG = open(log_path, 'w')
        LOG.write('Particles pdgs: ' + str(particles) + '\n')
        LOG.write('Selection: ' + selection + '\n')
        LOG.write('Number of tracks: ' + str(n_tracks) + '\n')
        LOG.flush()

    else:

        LOG = open(log_path, 'a')



    # Try create or open file with the READED data files.
    if not os.path.exists(readed_files_txt):

        READED = open(readed_files_txt, 'w')
        READED.write("")
        READED.close()
        READED = list(numpy.loadtxt(readed_files_txt, dtype='S', delimiter='\n', comments='#', ndmin=1))

    else:

        READED = list(numpy.loadtxt(readed_files_txt, dtype='S', delimiter='\n', comments='#', ndmin=1))


    # Count how many track have been taken
    try:
        data = pandas.read_csv(file_path, usecols=['MCParticleType'])

        numbers_per_particle = {}
        for pdg in particles:
            numbers_per_particle[pdg] = len(data[numpy.abs(data.MCParticleType.values) == int(pdg)])

        data = pandas.DataFrame()

    except:

        data = pandas.DataFrame()

        numbers_per_particle = {}
        for i in particles:
            numbers_per_particle[i] = 0


    for index in numbers_particles.index:

        success = 0
        while success != 1:
            try:
                file_http = numbers_particles.loc[index]['http']
                tree_name = numbers_particles.loc[index]['tree_name']

                # A file was readed before?
                if file_http in READED:
                    success = 1
                    continue

                branches = root_numpy.list_branches(file_http, treename=tree_name)
                branches = numpy.array(branches)

                data_array = root_numpy.root2array(filenames=file_http,
                                                   treename=tree_name,
                                                   branches=branches[branches != 'piplus_OWNPV_COV_'],
                                                   selection=selection)

                data = pandas.DataFrame(data=data_array, columns=branches[branches != 'piplus_OWNPV_COV_'])

                LOG.write(file_http + '\n')
                LOG.flush()

                data_iter = pandas.DataFrame(columns=branches[branches != 'piplus_OWNPV_COV_'])
                data_iter_index = []

                for one_particle in particles:

                    p_type = numpy.abs(data['MCParticleType'].values)
                    data_particle = data[p_type == int(one_particle)]

                    number = numbers_particles.loc[index][one_particle]
                    number_take = int(round(part[one_particle] * number))

                    data_particle_take_index, _ = train_test_split(data_particle.index,
                                                                   train_size=number_take,
                                                                   random_state=42)

                    data_iter_index += list(data_particle_take_index)
                    numbers_per_particle[one_particle] += number_take


                data_iter = data.loc[data_iter_index]

                if os.path.exists(file_path):
                    data_iter.to_csv(file_path, mode='a', header=False)
                else:
                    data_iter.to_csv(file_path, mode='a', header=True)

                del data_iter, data, data_array
                gc.collect()


                READED.append(file_http)
                numpy.array(READED).tofile(readed_files_txt, sep="\n")

                LOG.write('Tracks selected: ' + str(numbers_per_particle) + '\n')
                LOG.flush()

                success = 1

            except:

                LOG.write('Unexpected error \n')
                LOG.flush()

    return 1
