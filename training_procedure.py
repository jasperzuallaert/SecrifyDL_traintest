__author__ = 'jasper.zuallaert'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from input_manager import Dataset
import sys
import time

TRAINING_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 512
DROPOUT_RATE = 0.2
N_EPOCHS = 5

class TrainingProcedure:
    def __init__(self, network_object, train_dataset, valid_dataset, test_dataset, is_training):
        self.nn = network_object.getNetwork()
        self.X_placeholder = network_object.get_X_placeholder()
        self.seqlens_ph = network_object.getSeqLenPlaceholder()
        self.dropout_placeholder = network_object.getDropoutPlaceholder()
        self.Y_placeholder = tf.placeholder(tf.float32, [None, 1],name='Y_placeholder')
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.is_training = is_training

        self.predictions_logits = self.nn(self.X_placeholder,self.seqlens_ph)
        self.sigmoid_f = tf.sigmoid(self.predictions_logits)

        self.loss_f = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.Y_placeholder,logits=self.predictions_logits)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.train_op = self.optimizer.minimize(loss=self.loss_f,global_step=tf.train.get_or_create_global_step())
        self.total_parameters = self._print_num_params()

    # Prints the total number of trainable parameters
    # If this number does not exceed 5 million, and we are not running this class from a SingleTermWorkflow.py call,
    # the session (containing the network parameters) will be stored in the parameters/ directory)
    def _print_num_params(self):
        total_parameters = 0
        # iterating over all variables
        for variable in tf.trainable_variables():
            local_parameters = 1
            shape = variable.get_shape()  # getting shape of a variable
            for i in shape:
                local_parameters *= i.value  # mutiplying dimension values
            total_parameters += local_parameters
        print('This network has {} trainable parameters.'.format(total_parameters))
        return total_parameters

    def train_network(self,predictions_file,fold_number,dataset_name,timestamp):
        parameters_save_dest = f'parameters/{dataset_name}_{timestamp}_fold{fold_number}'
        ### create session ###
        print('session to be created')
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        self.sess = sess
        ### run initialization ###
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        self._print_output_classes(self.train_dataset, 'Training')
        self._print_output_classes(self.valid_dataset, 'Valid')
        self._print_output_classes(self.test_dataset, 'Test')

        print(' {:^5} | {:^14} | {:^14} | {:^14} | {:^14} | {:^12} | {:^12}'.format('epoch','train loss','valid loss','tr Fmax','va Fmax','total time','train time'))
        print('-{:-^6}+{:-^16}+{:-^16}+{:-^16}+{:-^16}+{:-^12}-{:-^13}-'.format('','','','','','','','','','','',''))

        ### Pre training, output ##
        best_valid_loss = 999999
        last_valid_loss = best_valid_loss
        going_up_for_epochs = 0
        not_improved_best_for_epochs = 0

        t1 = time.time()
        tr_loss, tr_Fmax, tr_avgPr, tr_avgSn = self._evaluate_set(self.train_dataset, VALIDATION_BATCH_SIZE)
        va_loss, va_Fmax, va_avgPr, va_avgSn = self._evaluate_set(self.valid_dataset, VALIDATION_BATCH_SIZE)

        print(' {:5d} |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |    {:4.2f}s     |   {:4.2f}s   '.format(0,tr_loss,va_loss,tr_Fmax,va_Fmax,time.time()-t1,0))

        ### train for each epoch ###
        for epoch in range(1,N_EPOCHS):
            sys.stdout.flush()
            epoch_start_time = time.time()

            epoch_finished = False
            trainstart = time.time()
            while not epoch_finished:
                ids, batch_x, lengths_x, batch_y, epoch_finished = self.train_dataset.next_batch(TRAINING_BATCH_SIZE)
                sess.run(self.train_op, feed_dict={self.X_placeholder: batch_x, self.Y_placeholder: batch_y, self.seqlens_ph:lengths_x, self.dropout_placeholder:DROPOUT_RATE, self.is_training:True})
            trainstop = time.time()

            tr_loss, tr_Fmax, tr_avgPr, tr_avgSn = self._evaluate_set(self.train_dataset, VALIDATION_BATCH_SIZE)
            va_loss, va_Fmax, va_avgPr, va_avgSn = self._evaluate_set(self.valid_dataset, VALIDATION_BATCH_SIZE)

            print_message = ''
            ### if new best validation result - store the parameters + generate predictions on test set ###
            if va_loss >= last_valid_loss:
                going_up_for_epochs += 1
                not_improved_best_for_epochs += 1
                if going_up_for_epochs > 3:
                    break
                if not_improved_best_for_epochs > 6:
                    break
            else:
                going_up_for_epochs = 0
                if va_loss < best_valid_loss:
                    not_improved_best_for_epochs = 0
                    best_valid_loss = va_loss
                    self._store_network_parameters(parameters_save_dest)
                    print_message = '-> New best valid.'
                else:
                    not_improved_best_for_epochs += 1
                    if not_improved_best_for_epochs > 6:
                        break
            last_valid_loss = va_loss

            print(' {:5d} |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {:4.2f}s     |   {:4.2f}s   {}'.format(epoch,tr_loss,va_loss,tr_Fmax,va_Fmax,time.time()-epoch_start_time,trainstop-trainstart,print_message))

        print("Finished")
        print('Parameters should\'ve been stored in {}'.format(parameters_save_dest))

        ### Generate predictions to show at the end of the file, using Evaluation.py  ###
        ### This is done based on the file with predictions that was written, so this ###
        ### could also be achieved by running Evaluation.py after this python program ###
        ### is finished.                                                              ###
        self._load_network_parameters(self.sess, parameters_save_dest)
        self._write_predictions(predictions_file)

        return sess

    # Generate the losses, f1 scores and other metrics for a given dataset
    def _evaluate_set(self, dataset: Dataset, batch_size, threshold_range = 20):
        losses = []
        all_preds = []
        all_labels = []
        F_per_thr = []
        pr_per_thr = []
        sn_per_thr = []

        ### go over each batch and store the losses ###
        batches_done = False
        while not batches_done:
            ids,batch_x, lengths_x, batch_y, epoch_finished = dataset.next_batch(batch_size)
            loss_batch = self.sess.run(self.loss_f, feed_dict={self.X_placeholder: batch_x, self.Y_placeholder: batch_y,self.seqlens_ph:lengths_x, self.is_training:False})
            preds_batch = self.sess.run(self.sigmoid_f, feed_dict={self.X_placeholder: batch_x, self.Y_placeholder: batch_y,self.seqlens_ph:lengths_x, self.is_training:False})
            losses.extend([loss_batch] * len(batch_x))
            all_preds.append(preds_batch)
            all_labels.append(batch_y)
            if epoch_finished:
                batches_done = True
        # return np.average(losses),-1,-1,-1
        ### at the desired epochs (currently: all), do the calculations ###
        all_preds = tf.concat(all_preds,axis=0)
        all_labels = tf.concat(all_labels,axis=0)

        ph_t = tf.placeholder(tf.float32)
        preds = tf.cast(tf.ceil(all_preds - ph_t),dtype=tf.int32)

        tp_f = tf.reduce_sum((all_labels + preds) // 2,axis=1)
        number_of_pos_f = tf.reduce_sum(all_labels,axis=1)
        predicted_pos_f = tf.reduce_sum(preds,axis=1)

        ### for every threshold, calculate pr, sn, fscore ###
        for t in range(threshold_range):
            threshold = t/threshold_range
            tp_res,n_of_pos_res,predicted_pos_res = self.sess.run([tp_f,number_of_pos_f,predicted_pos_f], feed_dict={ph_t:threshold})
            pr = sum(tp_res)/sum(predicted_pos_res) if sum(predicted_pos_res)>0 else 0.0
            sn = sum(tp_res)/sum(n_of_pos_res) if sum(n_of_pos_res)>0 else 0.0
            pr_per_thr.append(pr)
            sn_per_thr.append(sn)
            F_per_thr.append(2*pr*sn/(pr+sn) if pr+sn > 0 else 0.0)
        Fmax_index = int(np.argmax(F_per_thr))
        return np.average(losses), F_per_thr[Fmax_index], pr_per_thr[Fmax_index], sn_per_thr[Fmax_index]


    def _store_network_parameters(self, save_to_dir):
        try:
            saver = tf.train.Saver()
            if not os.path.exists(save_to_dir):
                os.makedirs(save_to_dir)
            saver.save(self.sess,save_to_dir+'/'+save_to_dir[save_to_dir.rfind('/')+1:])
        except Exception:
            print('Something went wrong while saving parameters! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(sys.exc_info())
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        pass

    def _load_network_parameters(self, sess, load_from_dir):
        filename = load_from_dir + '/' + load_from_dir[load_from_dir.rfind('/') + 1:]
        saver = tf.train.import_meta_graph(filename+'.meta')
        saver.restore(sess, tf.train.latest_checkpoint(load_from_dir))

    # Writes predictions to a file, to be evaluated by Evaluation.py afterwards
    def _write_predictions(self, predictions_file):
        batches_done = False
        while not batches_done:
            ids, batch_x, lengths_x, batch_y, epoch_finished = self.test_dataset.next_batch(VALIDATION_BATCH_SIZE)
            sigmoids = self.sess.run(self.sigmoid_f, feed_dict={self.X_placeholder: batch_x, self.Y_placeholder: batch_y,self.seqlens_ph:lengths_x, self.is_training:False})
            for id,p,c in zip(ids,sigmoids,batch_y):
                print(f'{id},{p[0]},{c[0]}',file=predictions_file)
            if epoch_finished:
                batches_done = True

    # Prints the information about the dataset in input
    # - dataset: an InputManager.Dataset object
    # - label: either 'Training', 'Valid', 'Test'
    def _print_output_classes(self, dataset, label):
        print(f'{label} set:')
        print(f'Number of positives: {dataset.get_positive_count()}')
        print(f'Number of negatives: {dataset.get_negative_count()}')

