from absl import logging
import enum
from datetime import datetime as dt
import functools
import numpy as np
import logging as logging_base
import operator
import os
import tensorflow as tf
import csv
import sys
sys.path.append('/Users/duanzy/Desktop/code1/2-prediction/da_rnn/metric/')
#from metric import metrics
import metrics
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_scaler import ZeroMaxScaler

class StrEnum(str, enum.Enum):
    pass

class RunnerPhase(StrEnum):
    TRAIN = 'train'
    VALIDATE = 'validate'
    PREDICT = 'predict'

class ModelRunner:
    def __init__(self,
                 model,
                 scaler,
                 flags,
                 save_path,
                 metrics=metrics.Metrics()):
        self.model_wrapper = ModelWrapper(model)
        self.model_checkpoint_path = flags.save_dir
        self.scaler = scaler
        self.lr = flags.learning_rate
        self.write_summary = flags.write_summary
        self.encoder_dim = flags.encoder_dim
        self.decoder_dim = flags.decoder_dim
        self.batch_size = flags.batch_size
        self.metrics = metrics
        self.save_path = save_path
        logging.get_absl_logger().addHandler(logging_base.StreamHandler())
        return

    @staticmethod
    def num_params():
        total_num = 0
        for var in tf.trainable_variables():
            shape = var.get_shape()
            total_num += functools.reduce(operator.mul, [dim.value for dim in shape], 1)
        return total_num

    def restore(self, restore_path):
        self.model_wrapper.restore(restore_path)

    def train(self,
              train_dataset,
              valid_dataset,
              test_dataset,
              max_epochs,
              valid_gap_epochs=1,
              auto_save=True):
        folder_path, model_path = self.make_dir()
        logging.get_absl_handler().use_absl_log_file('logs', folder_path)
        logging.info('Start training with max epochs {}'.format(max_epochs))
        logging.info('Model and logs saved in {}'.format(folder_path))
        logging.info('Number of trainable parameters - {}'.format(self.num_params()))
        if self.write_summary:
            self.model_wrapper.init_summary_writer(folder_path)
        best_eval_loss = float('inf')

        for i in range(1, max_epochs + 1):
            loss, _, _ = self.run_one_epoch(train_dataset, RunnerPhase.TRAIN, self.lr)
            loss_metrics = {'loss': np.mean(loss)}
            logging.info('Epoch {0} -- Training loss {1}'.format(i, loss_metrics['loss']))
            self.model_wrapper.write_summary(i, loss_metrics, RunnerPhase.TRAIN)

            if i % valid_gap_epochs == 0:
                '''
                csv_file = 'data/data_nasdaq/2.csv'
                raw_data = pd.read_csv(csv_file, header=0)
                data = np.array(raw_data.values)
                max_val, min_val = np.max(data), np.min(data)
                '''
                loss, prediction, _ = self.run_one_epoch(valid_dataset, RunnerPhase.VALIDATE, self.lr)
                loss_metrics = {'loss': np.mean(loss)}
                logging.info('Epoch {0} -- Validation loss {1}'.format(i, loss_metrics['loss']))
                self.model_wrapper.write_summary(i, loss_metrics, RunnerPhase.VALIDATE)
                predictions = valid_dataset.inverse_standard_scale(prediction, self.scaler[0], self.scaler[1])               
                '''
                min_value,max_value= min(prediction),max(prediction)
                prediction_temp = []
                for i in range(prediction.shape[0]-1):

                    inverse_prediction = (prediction[i,0]- min_value)/(max_value-min_value)*(max_val-min_val)+min_val
                    prediction_temp.append(inverse_prediction)
                '''
                self.write_train_predicted_data(self.save_path,i,predictions.flatten())

                if best_eval_loss > loss_metrics['loss']:
                    metrics = self.evaluate(test_dataset)
                    self.model_wrapper.write_summary(i, metrics, RunnerPhase.PREDICT)
                    best_eval_loss = loss_metrics['loss']
                    if auto_save:
                        self.model_wrapper.save(model_path)
        if not auto_save:
            self.model_wrapper.save(model_path)
        else:
            self.restore(model_path)
        logging.info('Training finished')

    def write_train_predicted_data(self,save_path,epoch,data):
        with open(os.path.join(save_path, 'predict_data_train.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(str(epoch))
            writer.writerow(data)
            csvfile.close()

    def write_test_predicted_data(self, save_path, data):
        with open(os.path.join(save_path, 'predict_data_test.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)
            csvfile.close()

    def evaluate(self, dataset, plot=False):
        logging.info('Start evaluation')
        loss, predictions, labels = self.run_one_epoch(dataset, RunnerPhase.PREDICT, self.lr)
        predictions = dataset.inverse_standard_scale(predictions, self.scaler[0], self.scaler[1])
        labels = dataset.inverse_standard_scale(labels, self.scaler[0], self.scaler[1])
        metrics_dict = self.metrics.get_metrics_dict(predictions, labels)
        eval_info = self.metrics.metrics_dict_to_str(metrics_dict)
        logging.info(eval_info)
        logging.info('Evaluation finished')
        if plot:
            self.plot_prediction(predictions, labels, num_plot=60)
        self.write_test_predicted_data(self.save_path,predictions.flatten())
        return metrics_dict

    def run_one_epoch(self, dataset, phase, lr):
        epoch_loss = []
        epoch_predictions = []
        epoch_labels = []
        total_batch = dataset.num_samples // self.batch_size
        for _ in range(total_batch):
            (input_x,), (input_label,), (input_x_exg,), (label,) = \
                zip(*dataset.next_batch(self.batch_size))
            loss, prediction = self.model_wrapper.run_batch((input_x,
                                                             input_label,
                                                             input_x_exg,
                                                             label),
                                                            lr,
                                                            phase=phase)
            epoch_loss.append(loss)
            epoch_predictions.append(prediction)
            epoch_labels.append(label)
        return np.array(epoch_loss), np.vstack(epoch_predictions), np.vstack(epoch_labels)

    def make_dir(self):
        folder_name = list()
        model_tags = {'lr': self.lr,
                      'encoder': self.encoder_dim,
                      'decoder': self.decoder_dim}
        for key, value in model_tags.items():
            folder_name.append('{}-{}'.format(key, value))
        folder_name = '_'.join(folder_name)
        current_time = dt.now().strftime('%Y%m%d-%H%M%S')
        folder_path = os.path.join(self.model_checkpoint_path,
                                   self.model_wrapper.__class__.__name__,
                                   folder_name,
                                   current_time)
        os.makedirs(folder_path)
        model_path = os.path.join(folder_path, 'saved_model')
        return folder_path, model_path

    @staticmethod
    def plot_prediction(predictions, labels, num_plot=200):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(labels[:num_plot], c='b', marker="^", ls='--', label='Ground True', fillstyle='none')
        ax.plot(predictions[:num_plot], c='k', marker="v", ls='-', label='DA-RNN')
        plt.legend(loc=2)
        plt.xlabel('Time axis')
        plt.show()
        return

class ModelWrapper:
    def __init__(self, model):
        self.model = model
        sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=sess_config)
        self.model.build()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.train_summary_writer = None
        self.valid_summary_writer = None
        self.test_summary_writer = None

    def save(self, checkpoint_path):
        self.saver.save(self.sess, checkpoint_path)

    def restore(self, checkpoint_path):
        self.saver.restore(self.sess, checkpoint_path)

    def init_summary_writer(self, root_dir):
        tf_board_dir = 'tfb_dir'
        folder = os.path.join(root_dir, tf_board_dir)
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(folder, 'train'), self.sess.graph)
        self.valid_summary_writer = tf.summary.FileWriter(os.path.join(folder, 'valid'))
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(folder, 'test'))

    def write_summary(self, epoch_num, kv_pairs, phase):
        if phase == RunnerPhase.TRAIN:
            summary_writer = self.train_summary_writer
        elif phase == RunnerPhase.VALIDATE:
            summary_writer = self.valid_summary_writer
        elif phase == RunnerPhase.PREDICT:
            summary_writer = self.test_summary_writer
        else:
            raise RuntimeError('Unknow phase: ' + phase)
        if summary_writer is None:
            return
        for key, value in kv_pairs.items():
            metrics = tf.Summary()
            metrics.value.add(tag=key, simple_value=value)
            summary_writer.add_summary(metrics, epoch_num)
        summary_writer.flush()

    def run_batch(self, batch_data, lr, phase):
        if phase == RunnerPhase.TRAIN:
            loss, prediction = self.model.train(self.sess, batch_data, lr=lr)
        else:
            loss, prediction = self.model.predict(self.sess, batch_data)

        return loss, prediction
