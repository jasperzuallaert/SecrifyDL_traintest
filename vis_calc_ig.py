__author__ = 'jasper.zuallaert'
import sys
import numpy as np

import input_manager as im
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def run_integrated_gradients_on_test_set(predictions_logits,
                                         sess,
                                         X_ph,
                                         seqlens_ph,
                                         dropout_ph,
                                         test_dataset,
                                         out_f,
                                         max_length):
    graph = tf.get_default_graph()

    ### tensor for gradient calculation on that embedding output
    gs = tf.gradients(predictions_logits, X_ph)
    num_integration_steps = 25
    epoch_finished = False
    while not epoch_finished:
        ids, batch_x, lengths_x, batch_y, epoch_finished = test_dataset.next_batch(512)
        lengths_x = [min(x,max_length) for x in lengths_x] # max 1002 by default!
        difference_part = batch_x / num_integration_steps

        ### Calculate the gradients for each step
        allNucs = np.argmax(batch_x,axis=-1)
        allClasses = [y[0] for y in batch_y]
        allSeqLens = lengths_x
        allValues = np.zeros(batch_x.shape, np.float32)
        allPreds = [p[0] for p in sess.run(tf.sigmoid(predictions_logits),feed_dict={X_ph: batch_x, seqlens_ph: lengths_x,dropout_ph: 0.0})]
        # allPreds = [p[0] for p in sess.run(predictions_logits,feed_dict={X_ph: batch_x, seqlens_ph: lengths_x,dropout_ph: 0.0})]
        allIDs = [x.rstrip().split('\t')[0] for x in ids]

        baseline = np.zeros(batch_x.shape)
        for step in range(1, num_integration_steps + 1):
            batch_x_for_this_step_1 = baseline + difference_part * (step - 1)
            batch_x_for_this_step_2 = baseline + difference_part * step

            all_gradients_1 = sess.run(gs, feed_dict={X_ph: batch_x_for_this_step_1, seqlens_ph: lengths_x,dropout_ph: 0.0})[0]
            all_gradients_2 = sess.run(gs, feed_dict={X_ph: batch_x_for_this_step_2, seqlens_ph: lengths_x,dropout_ph: 0.0})[0]

            allValues += (all_gradients_1 + all_gradients_2) / 2 * difference_part


        ### Generate outputs. Note that the sequence printed out could be truncated if the actual length surpasses the
        ### maximum length (1002 by default)
        for id, pred, seq, cl, seqlen, values in zip(allIDs, allPreds, allNucs, allClasses, allSeqLens, allValues):
            print('{},{},{},actual_length={}'.format(id,pred, cl, seqlen),file=out_f)
            print(','.join(['ACDEFGHIKLMNPQRSTVWY'[int(nuc)] for nuc in seq[:seqlen]]),file=out_f)
            print(','.join([str(score[int(nuc)]) for score, nuc in zip(values[:seqlen], seq[:seqlen])]),file=out_f)

# Function to call if we want to use IntegratedGradients.py from another file (such as SingleTermWorkflow.py)
# - For parameters, see the explanation for the function above
def run_from_session(sess, test_set, out_f, max_length):
    graph = tf.get_default_graph()
    prediction_logits = graph.get_tensor_by_name("my_logits/BiasAdd:0")
    X_placeholder = graph.get_tensor_by_name("X_placeholder:0")
    seqlen_ph = graph.get_tensor_by_name("seqlen_placeholder:0")
    dropout_ph = graph.get_tensor_by_name("dropout_placeholder:0")

    run_integrated_gradients_on_test_set(prediction_logits, sess, X_placeholder, seqlen_ph, dropout_ph, test_set, out_f, max_length)

# If called as a standalone python script, it should have the 5 arguments as stated below
# if len(sys.argv) != 7 and sys.argv[0] == 'IntegratedGradientsRunner.py':
#     print('Usage: python IntegratedGradientsRunner.py <term number> <parameter file> <train file> <test file> <use_reference> <output_file>')
# elif sys.argv[0] == 'IntegratedGradientsRunner.py':

# run: vis_calc_ig.py <parameter_file> <dataset_name>
# with <parameter_file> being the file with the parameters of the model, with its orginial name
# and <dataset_name> either 'sc' or 'pp'
if __name__ == '__main__':
    param_file = sys.argv[1].rstrip('/')
    fold_n = int(param_file[-1])
    dataset_name = sys.argv[2]

    from main import get_filenames_train_test, dataset_loc
    train_files, valid_file, test_file = get_filenames_train_test(dataset_loc[dataset_name])
    test_set = im.get_sequences(test_file)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    param_file = param_file
    param_file_full_name = param_file + '/' + param_file[param_file.rfind('/') + 1:]
    saver = tf.train.import_meta_graph(param_file_full_name + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint(param_file))

    run_from_session(sess, test_set, out_f=sys.stdout)

