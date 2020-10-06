import tensorflow as tf
from da_rnn.model.encoder_decoder import Encoder, Decoder


class DualStageRNN:
    def __init__(self, encoder_dim, decoder_dim, num_steps, num_series, use_cur_exg):
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.num_steps = num_steps
        self.num_series = num_series
        self.use_cur_exg = use_cur_exg

    def build(self):
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.num_steps, self.num_series])
        self.input_label = tf.placeholder(tf.float32, shape=[None, self.num_steps, 1])
        self.input_x_exg = tf.placeholder(tf.float32, shape=[None, 1, self.num_series])

        if self.use_cur_exg:
            self.input = tf.concat([self.input_x, self.input_x_exg], axis=1)
            self.encoder_steps = self.num_steps + 1
        else:
            self.input = self.input_x
            self.encoder_steps = self.num_steps

        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        self.lr = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        self.encoder = Encoder(self.encoder_dim, self.encoder_steps)
        self.decoder = Decoder(self.decoder_dim, self.num_steps)
        self.pred, self.loss = self.forward()
        self.opt_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        return

    def forward(self):
        encoder_states = self.encoder(self.input)
        pred = self.decoder(encoder_states, self.input_label)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(pred - self.labels), axis=-1))
        return pred, loss

    def train(self, sess, batch_data, lr):
        input_x, input_label, input_x_exg, label = batch_data
        feed_dict = {self.input_x: input_x,
                     self.input_label: input_label,
                     self.input_x_exg: input_x_exg,
                     self.labels: label,
                     self.lr: lr,
                     self.is_training: True}
        _, loss, prediction = sess.run([self.opt_op, self.loss, self.pred], feed_dict=feed_dict)
        return loss, prediction

    def predict(self, sess, batch_data):
        input_x, input_label, input_x_exg, label = batch_data
        feed_dict = {self.input_x: input_x,
                     self.input_label: input_label,
                     self.input_x_exg: input_x_exg,
                     self.labels: label,
                     self.is_training: False}
        loss, prediction = sess.run([self.loss, self.pred], feed_dict=feed_dict)
        return loss, prediction

