__author__ = 'jasper.zuallaert'
from network_topology_constructor import build_network_topology
from training_procedure import TrainingProcedure
from input_manager import get_sequences
import sys
from datetime import datetime
import os
import tensorflow as tf

MAX_LENGTH = 250

dataset_loc = {
    'pp': ('data/pp_{}.dat'),
    'sc': ('data/sc_{}.dat')
}

# The main script for running experiments. It combines calls to different python files.
def run(foldN, dataset_name, predictions_file, vis_file, varlen_red_strat, timestamp, visualize):
    if foldN > 0:
        tf.reset_default_graph()

    ### Read in training, validation and test sets ##
    train_files,valid_files,test_files = get_filenames_train_test(dataset_loc[dataset_name],foldN)

    train_set = get_sequences(datafiles=train_files,max_length=MAX_LENGTH)
    valid_set = get_sequences(datafiles=valid_files,max_length=MAX_LENGTH)
    test_set = get_sequences(datafiles=test_files,max_length=MAX_LENGTH)

    ### Build the topology ###
    nn, is_training = build_network_topology(varlen_red_strategy = varlen_red_strat, max_length=MAX_LENGTH)


    ### Trains the network (and at the end, stores predictions on the test set) ###
    tp = TrainingProcedure(network_object=nn,
                           train_dataset=train_set,
                           valid_dataset=valid_set,
                           test_dataset=test_set,
                           is_training = is_training)

    sess = tp.train_network(predictions_file, foldN, dataset_name, timestamp)
    if visualize:
        import vis_calc_ig
        vis_calc_ig.runFromSession(sess,test_set=test_set, out_f=vis_file)
    sess.close()

def get_filenames_train_test(datafile_template, foldN):
    train_files = {datafile_template.format(n) for n in range(10)}
    valid_files = {datafile_template.format(foldN)}
    train_files = train_files - valid_files
    test_files = ['data/robin_data_short.dat']
    return train_files, valid_files, test_files

# run as: main.py <dataset> <varlen_reduction_strategy>
# with <dataset> one of: sc, pp
# with <varlen_reduction_strategy> one of: global_maxp, k_maxp, gru, zero_padding
if __name__ == '__main__':
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    dataset_name = sys.argv[1]
    varlen_red_strat = sys.argv[2]
    assert dataset_name in ['pp','sc'], dataset_name+' not supported'
    assert varlen_red_strat in ['global_maxp', 'k_maxp', 'gru', 'zero_padding'], varlen_red_strat+' not supported'
    if not os.path.exists('predictions/'): os.mkdir('predictions')
    if not os.path.exists('parameters/'): os.mkdir('parameters')
    predictions_filename = 'predictions/{}_{}_{}.txt'.format(dataset_name,timestamp,'{}')
    vis_filename = 'visualizations/{}_{}_{}.txt'.format(dataset_name,timestamp,'{}')

    for foldN in range(10): # only do first
        run(foldN, dataset_name, open(predictions_filename.format(foldN),'w'), open(vis_filename.format(foldN),'w'), varlen_red_strat, timestamp, True)


