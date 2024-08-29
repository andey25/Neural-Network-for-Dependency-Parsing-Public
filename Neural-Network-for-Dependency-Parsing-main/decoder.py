import sys
import copy

import numpy as np
import torch

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos): # TODO Part 5

        
     state = State(range(1, len(words)))  
     state.stack.append(0)

     # Ensures the model is in evaluation mode
     self.model.eval()

     while state.buffer:
        # Prepares the input features using the feature extractor
        features = self.extractor.get_input_representation(words, pos, state)
        features_tensor = torch.tensor(np.array([features]), dtype=torch.long)

        
        # Predicts the next action
        with torch.no_grad():  
            logits = self.model(features_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        # Sorts the actions based on their probabilities
        sorted_actions = torch.argsort(probabilities, dim=1, descending=True)[0]

        # Selects the highest scoring permitted transition
        for action_idx in sorted_actions:
            action, label = self.output_labels[int(action_idx)]
            if action == 'shift' and (len(state.buffer) > 1 or len(state.stack) == 0):
                state.shift()
                break
            elif action == 'left_arc' and state.stack[-1] != 0:  # Ensures root is not the target
                state.left_arc(label)
                break
            elif action == 'right_arc' and len(state.stack) > 0:
                state.right_arc(label)
                break

     result = DependencyStructure()
     for p, c, r in state.deps:
        result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))

     return result



  



if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()


# Check the evaluate.py comments at the end for the results
