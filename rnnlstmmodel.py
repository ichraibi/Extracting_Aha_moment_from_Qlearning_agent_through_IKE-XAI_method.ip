#!/usr/local/bin/python
# -*- coding: utf-8 -*
import os
import sys

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM, Concatenate
from tensorflow.keras.callbacks import *


from sklearn.metrics import roc_auc_score

sys.path.insert(1, '../')
import tools

class Histories(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.aucs = []
		self.losses = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		y_pred = self.model.predict(self.validation_data[0])
		self.aucs.append(roc_auc_score(self.validation_data[1], y_pred))
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return


class LossHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.epoch = []
        self.history = {}

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return


class RnnLstmModel :
    def __init__(self,transition_dict,nb_LSTM_cell_per_hidden_layer,debug = False):

        print("\n\n-----------------------------------------------------------------------------------------------")
        print("STEP 5.1' - Create RNN Keras Model")
        print("-----------------------------------------------------------------------------------------------")

        input_sequence_size=nb_neurons_for_input_layer = nb_neurons_for_output_layer= len(transition_dict)

        self.rnn_model = self.build_keras_model_rnn(input_sequence_size,
                                                    nb_neurons_for_input_layer,
                                                    nb_LSTM_cell_per_hidden_layer,
                                                    nb_neurons_for_output_layer, debug)
        print("rnn_model :")
        print(self.rnn_model.summary())


    def build_keras_model_rnn(self, input_sequence_size, nb_neurons_for_input_layer, nb_LSTM_cell_per_hidden_layer,
                              nb_neurons_for_output_layer, debug=False):
        input_tensor = Input(shape=(None, input_sequence_size), name=None, dtype=None, sparse=False, tensor=None)
        input_layer_output = Dense(nb_neurons_for_input_layer, activation='sigmoid')(input_tensor)

        lstm_layer_output, lstm_state_h, lstm_state_c = LSTM(units=nb_LSTM_cell_per_hidden_layer,
                                                             return_sequences=True,
                                                             return_state=True)(input_layer_output)

        concat_input_and_hidden_layer_outputs = Concatenate()([input_layer_output, lstm_layer_output])

        output_layer = Dense(nb_neurons_for_output_layer, activation='sigmoid', name='output_layer')(
            concat_input_and_hidden_layer_outputs)

        rnn_model = Model(inputs=[input_tensor], outputs=[output_layer, lstm_state_h, lstm_state_c])

        return rnn_model

    def save_hidden_layer_states_and_predictions_into_files(self, dataset_name, predictions, state_h, state_c,
                                                            predictions_file,
                                                            state_h_file, state_c_file):
        print("\n       Nb predictions for ", dataset_name, " dataset: ", len(predictions))

        predictions_results = []
        for i in predictions:
            predictions_results.append(i)

        with open(predictions_file, 'w') as f:
            f.write(str(predictions_results))
        print("\n       Save lstm layer predictions for each symbol in sequences into file: ", predictions_file)

        h_state_results = []
        for i in state_h:
            h_state_results.append(i)

        with open(state_h_file, 'w') as f:
            f.write(str(h_state_results))
        print("       Save lstm layer hidden states into file: ", state_h_file)

        c_state_results = []
        for i in state_c:
            c_state_results.append(i)

        with open(state_c_file, 'w') as f:
            f.write(str(c_state_results))
        print("       Save lstm layer CEC into file: ", state_c_file)



    def load_existing_rnn_model(self, keras_model_rnn_backup_file):
        print("\n\n-----------------------------------------------------------------------------------------------")
        print("STEP 5 - Load Keras existing Model")
        print("-----------------------------------------------------------------------------------------------")

        self.rnn_model = keras.models.load_model(keras_model_rnn_backup_file)
        print(self.rnn_model.summary())

        return self.rnn_model




    def train_model(self,epochs,input_train_reshape,expected_output_train_reshape,input_train_val_reshape,expected_output_train_val_reshape,keras_model_rnn_backup_file,
                 history_losses_png_file, losses_by_epoch_png_file, accuracy_by_epoch_png_file,
                    result_dir,debug=False):

        print("\n\n-----------------------------------------------------------------------------------------------")
        print("STEP 5.2 - Train RNN Keras Model")
        print("-----------------------------------------------------------------------------------------------\n")

        self.rnn_model.compile(loss=['mean_squared_error', None, None], optimizer='adam', metrics=['accuracy'])

        history = LossHistory()

        print("\n epochs : ", epochs)

        if debug :
            print(" input_train_reshape.shape : ", input_train_reshape.shape,
              " - expected_output_train_reshape.shape : ", expected_output_train_reshape.shape,
              " - input_train_val_reshape.shape : ", input_train_val_reshape.shape,
              " - expected_output_train_val_reshape.shape : ", expected_output_train_val_reshape.shape,
              " - keras_model_rnn_backup_file : ", keras_model_rnn_backup_file)

        fit_result = self.rnn_model.fit(x=input_train_reshape,y=expected_output_train_reshape,
                                   validation_data=(input_train_val_reshape, expected_output_train_val_reshape),
                                   epochs=epochs,
                                   batch_size=1,
                                   verbose=1,
                                   callbacks=[history])

        if debug: print("\n fit_result.history : ", fit_result.history)
        print("Fit model on training data")

        # https://keras.io/getting-started/faq/ --> How can I save a Keras model?
        self.rnn_model.save(result_dir+keras_model_rnn_backup_file)
        print("5.3.1' Saving Keras Model RNN into file: ", result_dir+keras_model_rnn_backup_file)

        # uncomment this part bellow if you need to acess to this data
        # # History losses
        # np.save(result_dir+history_losses_backup_file, history.losses)
        # print("\n5.3.2' Saving history losses into file: ",result_dir+"history_losses.npy")
        #
        # # Train losses by epoch
        # np.save(result_dir+train_losses_by_epoch_backup_file, fit_result.history["loss"])
        # print("5.3.3' Saving train losses by epoch into file: ", result_dir+"train_losses_by_epoch.npy")
        #
        # # Validation losses by epoch
        # np.save(result_dir+validation_losses_by_epoch_backup_file, fit_result.history["val_loss"])
        # print("5.3.4' Saving validation losses by epoch into file: ", result_dir+"validation_losses_by_epoch.npy")
        #
        # # Train accuracy by epoch
        # np.save(result_dir+train_accuracy_by_epoch_backup_file, fit_result.history["output_layer_accuracy"])
        # print("5.3.5' Saving train accuracy by epoch into file: ", result_dir+"train_accuracy_by_epoch.npy")
        #
        # # Validation accuracy by epoch
        # np.save(result_dir+validation_accuracy_by_epoch_backup_file, fit_result.history["val_output_layer_accuracy"])
        # print("5.3.6' Saving validation accuracy by epoch into file: ", result_dir+"validation_accuracy_by_epoch.npy)

        print("\n\n-----------------------------------------------------------------------------------------------")
        print("STEP 5.4' - Create graphs for training RNN Keras Model")
        print("-----------------------------------------------------------------------------------------------\n")

        # history losses
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history.losses, label="losses")
        fig_title = "Training History Losses for " + str(epochs) + " epochs"
        plt.title(fig_title)
        plt.xlabel("Samples #")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig(result_dir+history_losses_png_file)
        print("5.4.1' Create graph for history losses into file: ", result_dir+history_losses_png_file)

        # loss by epoch (train and validation)
        N = np.arange(0, epochs)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, fit_result.history["loss"], label="train_loss")
        plt.plot(N, fit_result.history["val_loss"], label="val_loss")
        plt.title("Training and Validation Loss ")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(result_dir+losses_by_epoch_png_file)
        print("5.4.2' Create graph for losses by epoch (train and validation) into file: ", result_dir+losses_by_epoch_png_file)

        # accuracy by epoch (train and validation)
        N = np.arange(0, epochs)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, fit_result.history["output_layer_accuracy"], label="train_acc")
        plt.plot(N, fit_result.history["val_output_layer_accuracy"], label="val_acc")
        plt.title("Training and Validation Accuracy ")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(result_dir+accuracy_by_epoch_png_file)
        print("5.4.3' Create graph for accuracy by epoch (train and validation) into file: ",
              result_dir+accuracy_by_epoch_png_file)

    def evaluate_model (self,input_test_reshape, expected_output_test_reshape,input_train_reshape ,expected_output_train_reshape):
        print("\n\n-----------------------------------------------------------------------------------------------")
        print("STEP 6 - Evaluate training for RNN Keras Model for test dataset")
        print("-----------------------------------------------------------------------------------------------\n")

        results = self.rnn_model.evaluate(input_test_reshape, expected_output_test_reshape, batch_size=1, verbose=1)
        print("     RNN evaluation (test seq)      - Average [loss, accuracy] : ", results)

        results = self.rnn_model.evaluate(input_train_reshape, expected_output_train_reshape, batch_size=1, verbose=1)
        print("     RNN evaluation (train seq)     - Average [loss, accuracy] : ", results)

    def test_model(self,result_dir, input_test_reshape,transition_dict,dict_of_extracted_patterns,debug=False):
        print("\n\n-----------------------------------------------------------------------------------------------")
        print("STEP 7 - RNN Keras Model predictions for test dataset")
        print("-----------------------------------------------------------------------------------------------\n")

        input_test_reshape = tf.cast(input_test_reshape, tf.float32)

        model_predictions, hidden_pattern_h, states_c = self.rnn_model.predict(input_test_reshape)
        

        print("\n     Nb predictions for test dataset: ", len(model_predictions))
        file= result_dir+"dict_of_extracted_patterns.json"
        dict_of_extracted_patterns= tools.update_dict_of_extracted_patterns(dict_of_extracted_patterns, model_predictions, hidden_pattern_h, file, transition_dict, debug)
        
        return model_predictions,hidden_pattern_h, dict_of_extracted_patterns

    def export_hidden_patterns(self,model_predictions,lstm_state_h, lstm_state_c,dict_for_test,test_predictions_and_HS_json_file, test_sequences_json_file, rnn_weights_file, rnn_for_hidden_states_LSTM_layer_weights_file):
        print("\n\n-----------------------------------------------------------------------------------------------")
        print("STEP 8 - Get and Save data for rules extraction (hidden states, predictions, sequences, ...) - TEST")
        print("-----------------------------------------------------------------------------------------------")

        tools.save_hidden_layer_states_and_predictions_in_json_file(test_predictions_and_HS_json_file,
                                                                    model_predictions, lstm_state_h,
                                                                    lstm_state_c)

        tools.save_dict_in_json(dict_for_test, test_sequences_json_file)
        print("Saving sequences for rules extraction in file: ", test_sequences_json_file)

        with open(rnn_weights_file, 'w') as f:
            f.write(str(self.rnn_model.get_weights()))
        print("Save weights for hidden states extraction rnn model into file: ", rnn_weights_file)

        with open(rnn_for_hidden_states_LSTM_layer_weights_file, 'w') as f:
            f.write(str(self.rnn_model.layers[1].get_weights()[0]))
        print("\nSave LSTM layer weights for training rnn model into file: ",
              rnn_for_hidden_states_LSTM_layer_weights_file)

    def get_rnn_model(self):
        return self.rnn_model