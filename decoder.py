from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras
from numpy import reshape
import tensorflow as tf

from extract_training_data import FeatureExtractor, State
tf.compat.v1.disable_eager_execution()

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)

        while state.buffer:
            features = self.extractor.get_input_representation(words, pos, state)
            features = np.array(features)
            prediction = self.model.predict(features.reshape(1, -1))
            best_moves = np.argsort(prediction)
            best_moves = np.flip(best_moves[0])

            for i in range(0, len(best_moves)):
                move = self.output_labels[best_moves[i]]

                if (len(state.buffer) > 1 or (len(state.buffer) <= 1 and not state.stack)) and move[0] == 'shift':
                    state.shift()
                    break

                if state.stack:
                    if move[0] == 'left_arc' and state.stack[0] != '<ROOT>':
                        state.left_arc(move[1])
                        break

                    if move[0] == 'right_arc':
                        state.right_arc(move[1])
                        break

        result = DependencyStructure()
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, 'r')
        pos_vocab_f = open(POS_VOCAB_FILE, 'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2], 'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)