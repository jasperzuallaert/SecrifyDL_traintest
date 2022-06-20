__author__ = 'jasper.zuallaert'

import numpy as np
import tensorflow as tf

def build_network_topology(varlen_red_strategy, max_length):
    X_placeholder = tf.placeholder(tf.float32, [None, max_length, 20],name='X_placeholder')
    seqlen_ph = tf.placeholder(tf.int32, [None],name='seqlen_placeholder')
    dropout_placeholder = tf.placeholder(tf.float32,name='dropout_placeholder')
    is_training = tf.placeholder(tf.bool,name='is_train')

    return NetworkObject(
            build_my_network(varlen_red_strategy, dropout_placeholder),
            X_placeholder,
            seqlen_ph,
            dropout_placeholder
        ), is_training


# Prints the details of the neural network (layers and output shapes)
def print_neural_net(layers):
    print('Network information:')
    for l in layers:
        try:
            print('{:35s} -> {}'.format(l.name,l.shape))
        except AttributeError:
            pass

def build_my_network(varlen_red_strategy, dropout_placeholder):
    def network(X, seqlens):
        layers = []
        l = X
        layers.append(l)
        layers.append(tf.layers.conv1d(layers[-1], 100, 3, padding='same', activation=tf.nn.relu))
        layers.append(tf.layers.dropout(layers[-1], dropout_placeholder))
        layers.append(tf.layers.max_pooling1d(layers[-1], 4, 4))
        seqlens = seqlens // 4
        layers.append(tf.layers.conv1d(layers[-1], 100, 7, padding='same', activation=tf.nn.relu))
        layers.append(tf.layers.dropout(layers[-1], dropout_placeholder))
        layers.append(tf.layers.max_pooling1d(layers[-1], 4, 4))
        seqlens = seqlens // 4
        layers.append(tf.layers.conv1d(layers[-1], 100, 7, padding='same', activation=tf.nn.relu))
        layers.append(tf.layers.dropout(layers[-1], dropout_placeholder))

        if varlen_red_strategy == 'k_maxp':
            layers.append(tf.transpose(layers[-1], perm=[0, 2, 1]))
            values, _indices = tf.nn.top_k(layers[-1], k=5, sorted=True)
            layers.append(values)
        elif varlen_red_strategy == 'zero_padding':
            pass #do nothing special
        elif varlen_red_strategy == 'gru':
            layers.append(bidirec_gru_layer(layers[-1], seqlens, 512))
        elif varlen_red_strategy == 'global_maxp':
            layers.append(tf.layers.max_pooling1d(layers[-1], int(layers[-1].shape[1]), int(layers[-1].shape[1])))
        else:
            raise NotImplementedError(f'Reduction strategy "{varlen_red_strategy}" not supported')
        layers.append(tf.contrib.layers.flatten(layers[-1]))

        layers.append(tf.layers.dense(layers[-1],32))
        layers.append(tf.layers.dropout(layers[-1], dropout_placeholder))
        layers.append(tf.layers.dense(layers[-1],1,name='my_logits'))

        print_neural_net(layers)
        # The output layer here is returned as logits. Sigmoids are added in the TrainingProcedure.py file
        return layers[-1]

    return network

def bidirec_gru_layer(input, input_lengths, state_size):
    cellsFW = [tf.nn.rnn_cell.GRUCell(state_size)]
    cellsBW = [tf.nn.rnn_cell.GRUCell(state_size)]
    multiFW = tf.nn.rnn_cell.MultiRNNCell(cellsFW)
    multiBW = tf.nn.rnn_cell.MultiRNNCell(cellsBW)
    _, (stateFW,stateBW) = tf.nn.bidirectional_dynamic_rnn(multiFW, multiBW, input, dtype=tf.float32, sequence_length=input_lengths)
    lastCombined = tf.concat([stateFW[-1],stateBW[-1]],axis=1)
    return lastCombined

# Objects of this class hold a neural network tensor, as well as the placeholders used in that network
class NetworkObject:
    def __init__(self, network, X_placeholder, seqlen_ph, dropoutPlaceholder):
        self.network = network
        self.X_placeholder = X_placeholder
        self.seqlen_ph = seqlen_ph
        self.dropoutPlaceholder = dropoutPlaceholder

    def getNetwork(self):
        return self.network

    def getSeqLenPlaceholder(self):
        return self.seqlen_ph

    def get_X_placeholder(self):
        return self.X_placeholder

    def getDropoutPlaceholder(self):
        return self.dropoutPlaceholder


