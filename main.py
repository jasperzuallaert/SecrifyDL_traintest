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
    'pp': ('data/pp_{}.csv'),
    'sc': ('data/sc_{}.csv')
}

# The main script for running experiments. It combines calls to different python files.
def run(foldN, dataset_name, test_set_loc, predictions_file, vis_file, varlen_red_strat, timestamp, visualize):
    if foldN > 0:
        tf.reset_default_graph()

    ### Read in training, validation and test sets ##
    train_set = get_sequences(datafile=dataset_loc[dataset_name].format('train'),max_length=MAX_LENGTH)
    valid_set = get_sequences(datafile=dataset_loc[dataset_name].format('valid'),max_length=MAX_LENGTH)
    test_set = get_sequences(datafile=test_set_loc,max_length=MAX_LENGTH)

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

# run as: main.py <dataset> <varlen_reduction_strategy>
# with <dataset> one of: sc, pp
# with <varlen_reduction_strategy> one of: global_maxp, k_maxp, gru, zero_padding
if __name__ == '__main__':
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    training_set_name = sys.argv[1]
    varlen_red_strat = sys.argv[2]
    test_set_loc = sys.argv[3]

    assert training_set_name in ['pp', 'sc'], training_set_name + ' not supported'
    assert varlen_red_strat in ['global_maxp', 'k_maxp', 'gru', 'zero_padding'], varlen_red_strat+' not supported'
    if not os.path.exists('predictions/'): os.mkdir('predictions')
    if not os.path.exists('parameters/'): os.mkdir('parameters')
    if not os.path.exists('visualizations/'): os.mkdir('visualizations')
    predictions_filename = 'predictions/{}_{}_{}.txt'.format(training_set_name, timestamp, '{}')
    vis_filename = 'visualizations/{}_{}_{}.txt'.format(training_set_name, timestamp, '{}')

    for runN in range(10):
        run(runN, training_set_name, test_set_loc, open(predictions_filename.format(runN), 'w'),
            open(vis_filename.format(runN), 'w'), varlen_red_strat, timestamp, True)


