#!/usr/local/bin/python
# -*- coding: utf-8 -*
import tools
import numpy as np

class DataSetOfSequences:

    def __init__(self, sequences,transition_dict):
        self.train_set, self.train_val_set, self.test_set, self.dict_of_extracted_patterns = self.split_sequences_into_training_validation_test_datasets(sequences)

        self.input_train_reshape, self.expected_output_train_reshape, self.input_train_val_reshape, self.expected_output_train_val_reshape, self.input_test_reshape, self.expected_output_test_reshape, self.dict_for_test = self.transform_data_into_samples(transition_dict)


    def split_sequences_into_training_validation_test_datasets(self,sequences, debug=False):
        print("\n\n-----------------------------------------------------------------------------------------------")
        print("STEP 2 - Loading data into 3 datasets (train, validation and test)")
        print("-----------------------------------------------------------------------------------------------")

        print("\nInitial dataset length: ", len(sequences))

        train_set = sequences[:int(len(sequences) * .60)]
        rest_of_set = sequences[int(len(train_set)):]
        train_val_set = sequences[:int(len(rest_of_set) * .50)]
        test_set = rest_of_set[int(len(train_val_set)):]

        print(" Lenght of : ")
        print("- training dataset : ", len(train_set))
        print("- validation dataset  : ", len(train_val_set))
        print("- testing dataset  : ", len(test_set))

        dict_of_extracted_patterns = tools.initialize_dict_of_extracted_patterns(test_set, debug)
        print("Lenght of dict_of_extracted_patterns_of_test_step : ", len(dict_of_extracted_patterns))

        return train_set, train_val_set, test_set, dict_of_extracted_patterns


    def transform_data_into_samples(self, transition_dict, debug=False):
        print("\n\n-----------------------------------------------------------------------------------------------")
        print("STEP 3 - Transform data into samples")
        print("-----------------------------------------------------------------------------------------------")

        print("\n3.1 - Transform train data")
        [input_train, expected_output_train] = tools.transform_sequences_into_data_for_rnn(transition_dict, self.train_set,
                                                                                           False, debug)

        print("\n#####################################################################################################")
        print("\n3.2 - Transform train validation data")
        [input_train_val, expected_output_train_val] = tools.transform_sequences_into_data_for_rnn(transition_dict,
                                                                                                   self.train_val_set,
                                                                                                   False, debug)
        print(" self.input_train_val.shape : ", len(input_train_val))
        print(" self.expected_output_train_val.shape : ", len(expected_output_train_val))

        print("\n3.3 - Transform test data")
        print("\n#####################################################################################################")
        [input_test, expected_output_test, dict_for_test] = tools.transform_sequences_into_data_for_rnn(transition_dict,
                                                                                                        self.test_set,
                                                                                                        True, debug)

        print("Total input_train samples          : ", len(input_train))
        print("Total expected_output_train samples: ", len(expected_output_train))

        print("\nTotal input_validation samples          : ", len(input_train_val))
        print("Total expected_output_validation samples: ", len(expected_output_train_val))

        print("\nTotal input_test samples          : ", len(input_test))
        print("Total expected_output_test samples: ", len(expected_output_test))

        input_train_reshape, expected_output_train_reshape, input_train_val_reshape, expected_output_train_val_reshape, input_test_reshape, expected_output_test_reshape = self.reshape_data_for_keras_model(input_train, expected_output_train,input_train_val, expected_output_train_val,input_test, expected_output_test)

        print ("input_train_val_reshape, expected_output_train_val_reshape : ", input_train_val_reshape.shape, " - ", expected_output_train_val_reshape.shape)
        return  input_train_reshape, expected_output_train_reshape, input_train_val_reshape, expected_output_train_val_reshape, input_test_reshape, expected_output_test_reshape,dict_for_test


    def reshape_data_for_keras_model(self,input_train, expected_output_train,input_train_val, expected_output_train_val,input_test, expected_output_test, debug=False):
        print("\n\n-----------------------------------------------------------------------------------------------")
        print("STEP 4 - Reshape data for Keras Model")
        print("-----------------------------------------------------------------------------------------------")

        # TRAIN
        input_train = np.asarray(input_train)
        expected_output_train = np.asarray(expected_output_train)

        input_train_reshape = np.reshape(input_train, (input_train.shape[0], 1, input_train.shape[1]))
        expected_output_train_reshape = np.reshape(expected_output_train,
                                                   (expected_output_train.shape[0], 1, expected_output_train.shape[1]))

        # VALIDATION
        input_train_val = np.asarray(input_train_val)
        expected_output_train_val = np.asarray(expected_output_train_val)

        input_train_val_reshape = np.reshape(input_train_val, (input_train_val.shape[0], 1, input_train_val.shape[1]))
        expected_output_train_val_reshape = np.reshape(expected_output_train_val,
                                                       (expected_output_train_val.shape[0], 1,
                                                        expected_output_train_val.shape[1]))

        # TEST
        input_test = np.asarray(input_test)
        expected_output_test = np.asarray(expected_output_test)

        input_test_reshape = np.reshape(input_test, (input_test.shape[0], 1, input_test.shape[1]))
        expected_output_test_reshape = np.reshape(expected_output_test,
                                                  (expected_output_test.shape[0], 1, expected_output_test.shape[1]))

        print("\nReshape train data ...")
        if debug:
            print("\ninput_train_reshape.shape: ", input_train_reshape.shape)
            print("input_train_reshape: ", input_train_reshape)
            print("\n---------------")
            print("\nexpected_output_train_reshape.shape: ", expected_output_train_reshape.shape)
            print("expected_output_train_reshape: ", expected_output_train_reshape)
            print("\n-------------------------------------------\n")

        print("\nReshape train validation data ...")
        if debug:
            print("\ninput_train_val_reshape.shape: ", input_train_val_reshape.shape)
            print("input_train_val_reshape: ", input_train_val_reshape)
            print("\n---------------")
            print("\nexpected_output_train_val_reshape.shape: ", expected_output_train_val_reshape.shape)
            print("expected_output_train_val_reshape: ", expected_output_train_val_reshape)
            print("\n-------------------------------------------\n")

        print("\nReshape test data ...")
        if debug:
            print("\ninput_test_reshape.shape: ", input_test_reshape.shape)
            print("input_test_reshape: ", input_test_reshape)
            print("\n---------------")
            print("\nexpected_output_test_reshape.shape: ", expected_output_test_reshape.shape)
            print("expected_output_test_reshape: ", expected_output_test_reshape)


        return input_train_reshape, expected_output_train_reshape, input_train_val_reshape, expected_output_train_val_reshape, input_test_reshape, expected_output_test_reshape


    def get_input_train_reshape(self):
        return self.input_train_reshape

    def get_expected_output_train_reshape(self):
        return self.expected_output_train_reshape

    def get_input_train_val_reshape(self):
        return self.input_train_val_reshape

    def get_expected_output_train_val_reshape(self):
        return self.expected_output_train_val_reshape

    def get_input_test_reshape(self):
        return self.input_test_reshape

    def get_expected_output_test_reshape(self):
        return self.expected_output_test_reshape

    def get_dict_for_test(self):
        return self.dict_for_test

    def get_dict_of_extracted_patterns(self):
        return self.dict_of_extracted_patterns