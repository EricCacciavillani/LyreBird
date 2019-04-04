'''
This code is heavily based on the following repo:
https://github.com/scpark20/universal-music-translation

All I have done is reformat, comment, and restructure it for production like needs.
Please give the following credit or merit to authors.

They have done some incredible work and it was a genuine pleasure to read their paper and overall learn more.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
import librosa
from tqdm import tqdm
import datetime
import pretty_midi
import os
import shelve
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import soundfile as sf
import numpy as np

import sys
sys.path.append('..')

from Shared_Files.Custom_Exception import Model_Pathing_Error
from Shared_Files.Constants import *
from Shared_Files.Global_Util import *

class MusicTranslationModelGenerator():

    # Allows the gradient decent to be "Flipped" when doing backpropagating through a particular layer
    class __FlipGradientBuilder(object):
        def __init__(self):

            # Gives unique identifier with simple counter
            self.__num_calls = 0

        # Generate layer with custom gradient decent
        def __call__(self, x,
                     l=1.0):

            grad_name = "FlipGradient%d" % self.__num_calls

            @ops.RegisterGradient(grad_name)
            def _flip_gradients(op, grad):
                return [tf.negative(grad) * l]

            # Overrides default gradient decent for
            g = tf.get_default_graph()
            with g.gradient_override_map({"Identity": grad_name}):
                y = tf.identity(x)

            self.__num_calls += 1

            return y

    def __init__(self,
                 instr_wave_forms,
                 twilo_account=False):

        self.__instr_wave_forms = instr_wave_forms
        self.__twilo_account = twilo_account

        # ----------
        self.__instruments_list = list(self.__instr_wave_forms.keys())
        self.__INSTRUMENTS_NUM = len(self.__instruments_list)

    def create_model(self,
                     load_model_path=None,
                     training_sessions=0,
                     steps_per_training_session=0):


        # Resets the default graph stack/all global tf vars
        tf.reset_default_graph()

        #  --- Wavenet input  ---
        self.__x_holder = tf.placeholder(dtype=tf.float32, shape=[None, None])
        x_mulaw = self.__mulaw(self.__x_holder, UNIVERSAL_MUSIC_TRANSLATOR.MU)
        x_onehot_index = tf.clip_by_value(tf.cast((x_mulaw + 1.) * 0.5 * UNIVERSAL_MUSIC_TRANSLATOR.MU, tf.int32),
                                          0, UNIVERSAL_MUSIC_TRANSLATOR.MU - 1)

        # One hot encode tensor for each instrument naive wavenet
        x_onehot = tf.one_hot(x_onehot_index, depth=UNIVERSAL_MUSIC_TRANSLATOR.MU)

        # Label input
        label_holder = tf.placeholder(dtype=tf.int32, shape=())

        # --- Encoder layer ---

        # Define label encoder
        _, latents = self.__naive_wavenet(inputs=tf.expand_dims(self.__x_holder, axis=-1),
                                          condition=None, layers=9, h_filters=64,
                                          out_filters=UNIVERSAL_MUSIC_TRANSLATOR.LATENT_DIM,
                                          name='wavenet_encoder')

        # Downsample
        _, down_latents = self.__downsample(latents, UNIVERSAL_MUSIC_TRANSLATOR.POOL_SIZE)

        # Upsample
        self.__up_latents = self.__upsample(down_latents, tf.shape(self.__x_holder)[1], UNIVERSAL_MUSIC_TRANSLATOR.LATENT_DIM)

        '''
        DOMAIN CONFUSION LAYER
        '''
        flip_gradient = self.__FlipGradientBuilder()

        # Gradient reversal layer
        flipped_down_latents = flip_gradient(down_latents,
                                             l=1e-2)

        # domain predict
        label_predicts = self.__domain_confusion(flipped_down_latents, 3, self.__INSTRUMENTS_NUM, 128)
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

        pbar = tqdm(self.__instruments_list)
        for index, instr in enumerate(pbar):
            # decode
            dilation_sum, outputs = self.__naive_wavenet(inputs=x_onehot,
                                                         condition=self.__up_latents,
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
        self.__latents_holder = tf.placeholder(dtype=tf.float32,
                                               shape=[None, None, UNIVERSAL_MUSIC_TRANSLATOR.LATENT_DIM])

        self.__inference_sample_list = []

        pbar = tqdm(self.__instruments_list)
        for index, instr in enumerate(pbar):
            pbar.set_postfix_str(s=instr, refresh=True)
            # decode
            _, outputs = self.__naive_wavenet(inputs=x_onehot, condition=self.__latents_holder,
                                              layers=9, h_filters=64, out_filters=UNIVERSAL_MUSIC_TRANSLATOR.MU,
                                              name='wavenet_decoder_' + instr, reuse=True)
            outputs_probs = tf.nn.softmax(outputs)

            # sample from outputs
            dist = tf.distributions.Categorical(probs=outputs_probs[:, -1])
            sample = self.__inv_mulaw(tf.cast(dist.sample(), tf.float32) / UNIVERSAL_MUSIC_TRANSLATOR.MU * 2. - 1., UNIVERSAL_MUSIC_TRANSLATOR.MU)
            self.__inference_sample_list.append(sample)

        '''
        SESSION CREATE
        '''

        # Switch to true for verbose logs
        self.__sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        self.__sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if load_model_path:
            saver.restore(self.__sess, load_model_path)
            print("Restored model!")

        # # Restore variables from disk.

        print('Tensorflow graph created.')

        self.__sess.run(tf.global_variables_initializer())

        if training_sessions > 0:
            _epoch = 11
            neural_average_loss = 0.0
            max_lost_found = float("-inf")

            print("Begin training graph")

            recorded_losses = dict()

            for instr in self.__instruments_list:
                recorded_losses[instr] = list()
            recorded_losses["neural_average_loss"] = list()

            for session_num in range(training_sessions):

                pbar = tqdm(range(steps_per_training_session))
                for step in pbar:

                    for index, instr in enumerate(self.__instruments_list):

                        indexes = np.random.randint(0, self.__instr_wave_forms[instr].shape[0],
                                                    UNIVERSAL_MUSIC_TRANSLATOR.BATCH_SIZE)
                        augmented = []
                        for _wave in self.__instr_wave_forms[instr][indexes]:
                            augmented.append(self.__wave_augmentation(_wave))

                        augmented = np.stack(augmented, axis=0)

                        _, _loss = self.__sess.run([train_step, loss],
                                                   feed_dict={self.__x_holder: augmented,
                                                              label_holder: index})

                        neural_average_loss = 0.99 * neural_average_loss + 0.01 * _loss

                        recorded_losses[instr] += _loss
                        recorded_losses["neural_average_loss"] += neural_average_loss

                    pbar.set_postfix_str(s='neural_average_loss: {0}'.format(neural_average_loss), refresh=True)

                    self.__save(self.__sess, saver, ABS_PATHS.SAVED_MODELS_PATH_LYREBIRD_TN, step)

                    if max_lost_found < neural_average_loss:
                        max_lost_found = neural_average_loss
                        self.__save(self.__sess, saver, ABS_PATHS.SAVED_MODELS_PATH_LYREBIRD_TN_BEST, step)
                        print("hit")


                    # saver.save(sess, ABS_PATHS.SAVED_MODELS_PATH + "LyreBird_Model_" + str(datetime.datetime.now())[0:19])
                model_history_shelve = shelve.open(ABS_PATHS.SHELVES_PATH + SHELVE_NAMES.MODELS_HISTORY)

                # Shelf key found!
                if "recorded_losses" in model_history_shelve.keys():
                    # Append losses together for shelf
                    shelved_recorded_loses = model_history_shelve["recorded_losses"]
                    for instr, losses in shelved_recorded_loses.items():
                        shelved_recorded_loses[instr] += recorded_losses[instr]

                    model_history_shelve["recorded_losses"] = shelved_recorded_loses

                # Shelf key not found; init with current
                else:
                    model_history_shelve["recorded_losses"] = recorded_losses

                model_history_shelve.close()

                if self.__twilo_account:
                    send_sms_to_me("Model finished training session " + str(session_num) +" at " + str(datetime.datetime.now())[0:19] + " Avg Loss: " +
                                   str(neural_average_loss))
                self.generate_from_sample_music()

    def generate_from_sample_music(self,
                                   outputdir=ABS_PATHS.AUDIO_CACHE_PATH):
        src_instrument_index = "Piano"
        dest_instrument_index = 0

        index = np.random.randint(0, self.__instr_wave_forms[src_instrument_index].shape[0], 1)


        for key,val in self.__instr_wave_forms.items():
            print(key)
        _src = self.__instr_wave_forms[src_instrument_index][index]

        _latents = self.__sess.run(self.__up_latents, feed_dict={self.__x_holder: _src})

        librosa.output.write_wav(outputdir + 'Test_File_1.wav', _src[0], 8000, 1)

        _samples = np.zeros([1, 1024])
        _latents = np.concatenate([np.zeros([1, 1024, UNIVERSAL_MUSIC_TRANSLATOR.LATENT_DIM]), _latents], axis=1)
        for i in tqdm(range(10000)):
            _inference_sample_list = self.__sess.run(self.__inference_sample_list, feed_dict={self.__x_holder: _samples[:, -1024:],
                                                                                              self.__latents_holder: _latents[:,
                                                                                                i:i + 1024]})
            _samples = np.concatenate([_samples, np.expand_dims(_inference_sample_list[dest_instrument_index], axis=0)],
                                      axis=-1)

        librosa.output.write_wav(outputdir + 'Test_File_2.wav', _src[0], 8000, 1)
        librosa.output.write_wav(outputdir + 'Test_File_3.wav', _samples[0], 8000, 1)


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
            outputs = tf.layers.conv1d(inputs=outputs, filters=h_filters,
                                       kernel_size=2,
                                       dilation_rate=1, activation=tf.nn.elu)

        outputs = tf.layers.dense(outputs, domain_num, activation=tf.nn.tanh)
        outputs = tf.layers.dense(outputs, domain_num)
        return outputs

    ### Notes augmetation
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
        return self.__pitch_shift(inputs, start_index, end_index, n_steps)

    def __save(self,session, saver, checkpoint_dir, step):
        dir = os.path.join(checkpoint_dir, "model.ckpt")
        saver.save(session, dir, global_step=step)