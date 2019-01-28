'''
This code is heavily based on the following repo:
https://github.com/scpark20/universal-music-translation

All I have done is reformat and restructure it for production like needs;
I have also attempted to make make slight modifications on the model and functionality of their design.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
import librosa
from tqdm import tqdm

import numpy as np

import sys
sys.path.append('..')

from Shared_Files.Custom_Exception import Model_Pathing_Error
from Shared_Files.Constants import *

class MusicTranslation():

    class __FlipGradientBuilder(object):
        def __init__(self):
            self.num_calls = 0

        def __call__(self, x, l=1.0):
            grad_name = "FlipGradient%d" % self.num_calls

            @ops.RegisterGradient(grad_name)
            def _flip_gradients(op, grad):
                return [tf.negative(grad) * l]

            g = tf.get_default_graph()
            with g.gradient_override_map({"Identity": grad_name}):
                y = tf.identity(x)

            self.num_calls += 1
            return y

    def __init__(self,
                 pre_processor_obj,
                 twilo_account=False):
        """
        :param pre_processor_obj:
            Must be a midi pre-processor obj to extract data for training the model
        :param twilo_account
            If True then assumes a twilo account is properly set up and attempts to send sms messages;
        """

        # Gets instr_note_waves from the pre-processor
        self.__instr_wave_forms = pre_processor_obj.return_instr_wave_forms()

        # ----------
        self.__twilo_account = twilo_account
        self.__INSTRUMENTS_NUM = len(self.__instr_wave_forms.keys())
        self.__sess = None

        for instr, waves in self.__instr_wave_forms.items():
            print("instr:{0} Matrix_Shape: {1}".format(instr, waves.shape))

    def create_model(self, model_path):

        # Resets the default graph stack and all global default graph.
        tf.reset_default_graph()

        # Define wave input
        x_holder = tf.placeholder(dtype=tf.float32, shape=[None, None])
        print(type(x_holder))
        x_mulaw = self.__mulaw(x_holder, UNIVERSAL_MUSIC_TRANSLATOR.MU)
        print(type(x_mulaw))
        x_onehot_index = tf.clip_by_value(tf.cast((x_mulaw + 1.) * 0.5 * UNIVERSAL_MUSIC_TRANSLATOR.MU, tf.int32),
                                          0, UNIVERSAL_MUSIC_TRANSLATOR.MU - 1)
        x_onehot = tf.one_hot(x_onehot_index, depth=UNIVERSAL_MUSIC_TRANSLATOR.MU)

        # Label input
        label_holder = tf.placeholder(dtype=tf.int32, shape=())

        '''
        ENCODER LAYER
        '''

        # encode
        _, latents = self.__naive_wavenet(inputs=tf.expand_dims(x_holder, axis=-1),
                                   condition=None,
                                   layers=9, h_filters=64, out_filters=UNIVERSAL_MUSIC_TRANSLATOR.LATENT_DIM,
                                   name='wavenet_encoder')

        # downsample
        _, down_latents = self.__downsample(latents, UNIVERSAL_MUSIC_TRANSLATOR.POOL_SIZE)

        # upsample
        up_latents = self.__upsample(down_latents, tf.shape(x_holder)[1], UNIVERSAL_MUSIC_TRANSLATOR.LATENT_DIM)

        '''
        DOMAIN CONFUSION LAYER
        '''
        flip_gradient = self.__FlipGradientBuilder()

        # gradient reversal layer
        flipped_down_latents = flip_gradient(down_latents, l=1e-2)
        # flipped_down_latents = down_latents

        # domain predict
        label_predicts = self.__domain_confusion(flipped_down_latents, 3, self.__INSTRUMENTS_NUM,
                                          128)
        label_predicts = tf.reduce_mean(label_predicts, axis=1)
        label_predicts_prob = tf.nn.softmax(label_predicts)
        label_tiled = tf.tile(tf.expand_dims(label_holder, axis=0),
                              [tf.shape(label_predicts)[0]])

        # loss
        domain_confusion_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=label_tiled, logits=label_predicts)

        '''
        DECODER LAYER for traininng
        '''
        decode_losses = []
        samples_list = []

        print("DECODER LAYER for traininng")
        instrument_list = list(self.__instr_wave_forms.keys())

        pbar = tqdm(instrument_list)
        for instr in pbar:
            # decode
            dilation_sum, outputs = self.__naive_wavenet(inputs=x_onehot,
                                                  condition=up_latents,
                                                  layers=9, h_filters=64,
                                                  out_filters=UNIVERSAL_MUSIC_TRANSLATOR.MU,
                                                  name='wavenet_decoder_' + instr)

            pbar.set_postfix_str(s=instr, refresh=True)
            outputs_probs = tf.nn.softmax(outputs)

            # sample from outputs
            dist = tf.distributions.Categorical(probs=outputs_probs)
            samples = self.__inv_mulaw(tf.cast(dist.sample(), tf.float32) / UNIVERSAL_MUSIC_TRANSLATOR.MU * 2. - 1., UNIVERSAL_MUSIC_TRANSLATOR.MU)

            samples_list.append(samples)

            # loss
            decode_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=x_onehot_index[:, dilation_sum + 1:],
                logits=outputs[:, dilation_sum:-1])
            decode_loss = tf.reduce_mean(decode_loss)
            decode_losses.append(decode_loss)

        decode_losses = tf.stack(decode_losses, axis=0) * tf.one_hot(label_holder,
                                                                     depth=self.__INSTRUMENTS_NUM)
        decode_losses = tf.reduce_mean(decode_losses)
        loss = decode_losses + domain_confusion_loss
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

        '''
        DECODER LAYER for inference
        '''

        print("Define placeholder.")
        # input for decoder
        latents_holder = tf.placeholder(dtype=tf.float32,
                                        shape=[None, None, UNIVERSAL_MUSIC_TRANSLATOR.LATENT_DIM])

        inference_sample_list = []

        pbar = tqdm(instrument_list)
        for instr in pbar:
            pbar.set_postfix_str(s=instr, refresh=True)
            # decode
            _, outputs = self.__naive_wavenet(inputs=x_onehot, condition=latents_holder,
                                              layers=9, h_filters=64, out_filters=UNIVERSAL_MUSIC_TRANSLATOR.MU,
                                              name='wavenet_decoder_' + instr, reuse=True)
            outputs_probs = tf.nn.softmax(outputs)

            # sample from outputs
            dist = tf.distributions.Categorical(probs=outputs_probs[:, -1])
            sample = self.__inv_mulaw(tf.cast(dist.sample(), tf.float32) / UNIVERSAL_MUSIC_TRANSLATOR.MU * 2. - 1., UNIVERSAL_MUSIC_TRANSLATOR.MU)
            inference_sample_list.append(sample)

        '''
        SESSION CREATE
        '''

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())
        print(tf.global_variables_initializer())

        saver = tf.train.Saver()

        saver.save(sess, '/home/eric/Desktop/LyreBird/Data_Dump/Saved_Models/LyreBird_Music_Translation_1')


        # Restore variables from disk.
        saver.restore(sess, "/home/eric/Desktop/LyreBird/Data_Dump/Saved_Models/LyreBird_Music_Translation_1")
        print("Model restored.")

        print('Tensorflow graph created.')

        self.__sess = sess
        #
    def train_model(self,
                    model_path,
                    generate_graphs_path=None):

        self.__sess.run(tf.global_variables_initializer())

        _epoch = 11
        average_loss = 0.0

        print("While loop TRAINING")
        while True:

            ### TRAINING
            for i in tqdm(range(STEPS_PER_EPOCH)):
                for instrument_index in range(INSTRUMENTS_NUM):
                    batch_size = 3
                    indexes = np.random.randint(0, inst_waves_list[
                        instrument_index].shape[0], batch_size)
                    augmented = []
                    for _wave in inst_waves_list[instrument_index][indexes]:
                        augmented.append(self.__wave_augmentation(_wave))

                    augmented = np.stack(augmented, axis=0)
                    _, _loss = sess.run([train_step, loss],
                                        feed_dict={x_holder: augmented,
                                                   label_holder: instrument_index})

                    average_loss = 0.99 * average_loss + 0.01 * _loss
                    print('step : ', i, 'instrument : ', instrument_index, 'loss : ',
                          _loss, 'average_loss : ', average_loss)

    # Mulaw is a way of compressing for audio waves
    def __mulaw(self, x, MU):
        return tf.sign(x) * tf.log(1. + MU * tf.abs(x)) / \
               tf.log(1. + MU)


    def __inv_mulaw(self, x, MU):
        return tf.sign(x) * (1. / UNIVERSAL_MUSIC_TRANSLATOR.MU) * (tf.pow(1.
                                                                           + MU,
                                                                           tf.abs(x)) - 1.)

    def __naive_wavenet(self, inputs, condition, layers, h_filters, out_filters,
                      name='naive_wavenet', reuse=False):

        with tf.variable_scope(name, reuse=reuse):

            outputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])
            outputs = tf.layers.conv1d(inputs=outputs, filters=h_filters,
                                       kernel_size=2, dilation_rate=1,
                                       use_bias=False)
            dilation_sum = 1
            skips = []

            for layer in range(layers):
                dilation = 2 ** layer
                dilation_sum += dilation
                layer_outputs = tf.pad(outputs, [[0, 0], [dilation, 0], [0, 0]])
                filter_outputs = tf.layers.conv1d(inputs=layer_outputs,
                                                  filters=h_filters,
                                                  kernel_size=2,
                                                  dilation_rate=dilation,
                                                  use_bias=False)
                gate_outputs = tf.layers.conv1d(inputs=layer_outputs,
                                                filters=h_filters,
                                                kernel_size=2,
                                                dilation_rate=dilation,
                                                use_bias=False)
                if condition is not None:
                    filter_condition = tf.layers.dense(condition, h_filters)
                    gate_condition = tf.layers.dense(condition, h_filters)
                else:
                    filter_condition = 0
                    gate_condition = 0

                layer_outputs = tf.nn.tanh(filter_outputs + filter_condition) * \
                                tf.nn.sigmoid(gate_outputs + gate_condition)

                residual = tf.layers.dense(layer_outputs, h_filters)
                outputs += residual

                skip = tf.layers.dense(layer_outputs, h_filters)
                skips.append(skip)

            outputs = tf.nn.relu(sum(skips))
            outputs = tf.layers.dense(outputs, out_filters, activation=tf.nn.relu)
            outputs = tf.layers.dense(outputs, out_filters, activation=None)

        return dilation_sum, outputs

    def __downsample(self, inputs, pool_size):
        outputs = tf.layers.average_pooling1d(inputs=inputs,
                                              pool_size=pool_size,
                                              strides=pool_size)
        pad_size = 0
        return pad_size, outputs

    def __upsample(self, inputs, output_size, channel):
        outputs = tf.expand_dims(inputs, axis=1)
        outputs = tf.image.resize_nearest_neighbor(outputs, [1, output_size])
        outputs = tf.squeeze(outputs, axis=1)
        outputs = tf.reshape(outputs,
                             [tf.shape(outputs)[0], tf.shape(outputs)[1], channel])
        return outputs[:, -output_size:]

    def __domain_confusion(self, inputs, layers, domain_num, h_filters):
        outputs = inputs
        for layer in range(layers):
            dilation = 2 ** layers
            outputs = tf.layers.conv1d(inputs=outputs, filters=h_filters,
                                       kernel_size=2,
                                       dilation_rate=1, activation=tf.nn.elu)

        outputs = tf.layers.dense(outputs, domain_num, activation=tf.nn.tanh)
        outputs = tf.layers.dense(outputs, domain_num)
        return outputs

    ### data augmetation
    def __pitch_shift(self, inputs, start_index, end_index, n_steps):
        shifted = librosa.effects.pitch_shift(inputs[start_index:end_index], 8000,
                                              n_steps)
        outputs = np.concatenate(
            [inputs[:start_index], shifted, inputs[end_index:]], axis=0)
        return outputs

    def __wave_augmentation(self, inputs):

        length = np.random.randint(2000, 4000, 1)[0]
        start_index = np.random.randint(0, len(inputs) - length, 1)[0]
        end_index = start_index + length
        n_steps = float(np.random.ranf(1)[0] - 0.5)
        return self.pitch_shift(inputs, start_index, end_index, n_steps)
