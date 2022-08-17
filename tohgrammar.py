#!/usr/local/bin/python
# -*- coding: utf-8 -*

import numpy as np
import random

class TohGrammar :
    def __init__(self, symbols, dict, debug = False):
        self.symbols_list = ["B"]+symbols+["E"]
        self.symbols_dict = self.create_dictLettersWithBinaryValues(debug)
        self.grammar_rules = dict
        self.starting_index = sorted(dict.keys(), key=int)[0]
        self.ending_index = sorted(dict.keys(), key=int)[-1]
        print("starting index : ", self.starting_index, " - ending index ", self.ending_index)
        if debug:
            for k, v in self.symbols_dict.items():
                print(k, " : ", v)

    def getLetterByBinaryValue(self, ABinaryValue):
        theLetter = ""
        for KeyLetter, BinaryLetter in self.symbols_dict.items():
            if np.array_equal(BinaryLetter, ABinaryValue) == True:
                theLetter = KeyLetter
                break

        return theLetter


    def create_dictLettersWithBinaryValues(self, debug = False):

        LettersDict = {}
        i = 0
        while (i < len(self.symbols_list)):
            currentLetter = self.symbols_list[i]

            ABinaryLetter = []
            for aletter in self.symbols_list:

                if aletter == currentLetter:
                    ABinaryLetter.append(1)
                else:
                    ABinaryLetter.append(0)

            LettersDict[currentLetter] = ABinaryLetter

            i = i + 1


        return LettersDict

    def create_sequences_hanoi(self,n, debug=False):
        dict= {}
        sequences = []
        LenSequences = []
        for i in range(n):
            [sequence, sequence_of_indexes, sequence_of_labels] = self.create_one_sequence_Hanoi()
            if sequence in sequences:
                while sequence in sequences:
                    [sequence, sequence_of_indexes, sequence_of_labels] = self.create_one_sequence_Hanoi()

            if debug:
                print(sequence)

            sequences.append(sequence)
            LenSequences.append(len(sequence))
            dict[sequence] ={"sequence" : sequence, "sequence_of_indexes":sequence_of_indexes, "sequence_of_labels": sequence_of_labels}

        print ("Generation of ", len(sequences), " sequences done !")

        print (" --> The average len is : ", self.mean(LenSequences))
        print (" --> The standard deviation is : ", self.pstdev(LenSequences))

        return sequences

    def create_one_sequence_Hanoi(self):

        #print ("self.grammar_rules.keys() : ", self.grammar_rules.keys())
        sequence = 'B'
        index = self.starting_index

        sequence_of_indexes = str(index)
        sequence_of_labels = self.grammar_rules[index]["label"]

        while index !=  self.ending_index :
            choices = self.grammar_rules[index]["transitions"]
            choice = np.random.randint(0, len(choices))
            token, index = choices[choice]
            #print ("token : ", token, " - index ", index )
            sequence += ";"+token

            if index != self.ending_index:
                sequence_of_indexes += ";" + str(index)
                sequence_of_labels += ";"+self.grammar_rules[index]["label"]


        sequence+=";E"
        sequence_of_labels += ";333"
       
        assert((sequence.count(";")-2)==sequence_of_indexes.count(";")==sequence_of_labels.count(";")-1)


        return [sequence,sequence_of_indexes, sequence_of_labels]


    def mean(self, data):
        """Return the sample arithmetic mean of data."""
        n = len(data)
        if n < 1:
            raise ValueError('mean requires at least one data point')
        return sum(data) / float(n)  # in Python 2 use sum(data)/float(n)

    def _ss(self, data):
        """Return sum of square deviations of sequence data."""
        c = self.mean(data)
        ss = sum((x - c) ** 2 for x in data)
        return ss

    def pstdev(self, data):
        """Calculates the population standard deviation."""
        n = len(data)
        if n < 2:
            raise ValueError('variance requires at least two data points')
        ss = self._ss(data)
        pvar = ss / n  # the population variance
        return pvar ** 0.5

    def get_symbols_dict(self):
        return self.symbols_dict

    