import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
#from keras.layers import Bidirectional
#from keras import initializers, regularizers, constraints


class Attention(keras.Model):
    def __init__(self, input_dim, var_scope, reuse=tf.AUTO_REUSE):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        with tf.variable_scope(var_scope, reuse=reuse):
            self.attention_w = layers.Dense(self.input_dim, name='W')
            self.attention_u = layers.Dense(self.input_dim, name='U')
            self.attention_v = layers.Dense(1, name='V')

    def call(self, input_x, prev_state_tuple):
        prev_hidden_state, prev_cell_state = prev_state_tuple
        concat_state = tf.expand_dims(tf.concat([prev_hidden_state, prev_cell_state], axis=-1),
                                      axis=1)

        score_ = self.attention_w(concat_state) + self.attention_u(input_x)
        score = self.attention_v(tf.nn.tanh(score_))
        weight = tf.squeeze(tf.nn.softmax(score, axis=1), axis=-1)
        return weight


class LSTMCell(keras.Model):
    def __init__(self, hidden_dim):
        super(LSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_fc = layers.Dense(self.hidden_dim)

    def call(self, input_x, prev_state_tuple):
        hidden_state, cell_state = prev_state_tuple
        concat_input = tf.concat([hidden_state, input_x], axis=-1)
        concat_input_tiled = tf.tile(concat_input, [4, 1])
        forget_, input_, output_, cell_bar = tf.split(self.layer_fc(concat_input_tiled),
                                                      axis=0,
                                                      num_or_size_splits=4)

        cell_state = tf.nn.sigmoid(forget_) * cell_state + \
                     tf.nn.sigmoid(input_) * tf.nn.tanh(cell_bar)
        hidden_state = tf.nn.sigmoid(output_) * tf.nn.tanh(cell_state)
        return (hidden_state, cell_state)

class Encoder(keras.Model):
    def __init__(self, encoder_dim, num_steps):
        super(Encoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_layer = Attention(num_steps, var_scope='input_attention')
        self.lstm_cell = LSTMCell(encoder_dim)

    def call(self, inputs):
        def one_step(prev_state_tuple, current_input):
            inputs_scan = tf.transpose(inputs, perm=[0, 2, 1])
            weight = self.attention_layer(inputs_scan, prev_state_tuple)
            weighted_current_input = weight * current_input
            return self.lstm_cell(weighted_current_input, prev_state_tuple)

        self.batch_size = tf.shape(inputs)[0]
        self.num_steps = inputs.get_shape().as_list()[1]
        self.init_hidden_state = tf.random_normal([self.batch_size, self.encoder_dim])
        self.init_cell_state = tf.random_normal([self.batch_size, self.encoder_dim])
        inputs_ = tf.transpose(inputs, perm=[1, 0, 2])

        state_tuple = tf.scan(one_step,
                              elems=inputs_,
                              initializer=(self.init_hidden_state,
                                           self.init_cell_state))
        all_hidden_state = tf.transpose(state_tuple[0], perm=[1, 0, 2])
        return all_hidden_state


class Decoder(keras.Model):
    def __init__(self, decoder_dim, num_steps,
                 f_regularizer=None, b_regularizer=None,
                 f_constraint=None, b_constraint=None,):

        super(Decoder, self).__init__()
        self.decoder_dim = decoder_dim
        self.attention_layer = Attention(num_steps, var_scope='temporal_attention')
        self.lstm_cell = LSTMCell(decoder_dim)
        self.layer_fc_context = layers.Dense(1)
        self.layer_prediction_fc_1 = layers.Dense(decoder_dim)
        self.layer_prediction_fc_2 = layers.Dense(1)

        '''
        self.f_regularizer = regularizers.get(f_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.f_constraint = constraints.get(f_constraint)
        self.b_constraint = constraints.get(b_constraint)            

        lstm_fw_cell = rnn_cell.BasicLSTMCell(self.n_hidden_text, forget_bias=1.0, state_is_tuple=True) 
        lstm_bw_cell = rnn_cell.BasicLSTMCell(self.n_hidden_text, forget_bias=1.0, state_is_tuple=True)  
        lstm_fw_cell = rnn_cell.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        lstm_bw_cell = rnn_cell.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1)))'''

    def call(self, encoder_states, labels):
        def one_step(accumulator, current_label):
            prev_state_tuple, context = accumulator
            weight = self.attention_layer(encoder_states, prev_state_tuple)
            context = tf.reduce_sum(tf.expand_dims(weight, axis=-1) * encoder_states, axis=1)
            y_tilde = self.layer_fc_context(tf.concat([current_label, context], axis=-1))
            return self.lstm_cell(y_tilde, prev_state_tuple), context

        self.batch_size = tf.shape(encoder_states)[0]
        self.num_steps = encoder_states.get_shape().as_list()[1]
        self.encoder_dim = encoder_states.get_shape().as_list()[-1]

        init_hidden_state = tf.random_normal([self.batch_size, self.decoder_dim])
        init_cell_state = tf.random_normal([self.batch_size, self.decoder_dim])
        init_context = tf.random_normal([self.batch_size, self.encoder_dim])

        inputs_ = tf.transpose(encoder_states, perm=[1, 0, 2])
        state_tuple, all_context = tf.scan(one_step,
                                           elems=inputs_,
                                           initializer=((init_hidden_state,
                                                        init_cell_state),
                                                        init_context))
        all_hidden_state = tf.transpose(state_tuple[0], perm=[1, 0, 2])
        all_context = tf.transpose(all_context, perm=[1, 0, 2])
        last_hidden_state = all_hidden_state[:, -1, :]
        last_context = all_context[:, -1, :]
        pred_ = self.layer_prediction_fc_1(tf.concat([last_hidden_state, last_context], axis=-1))
        pred = self.layer_prediction_fc_2(pred_)
        return pred

