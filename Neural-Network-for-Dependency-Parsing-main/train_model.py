import sys
import numpy as np
import torch

from torch.nn import Module, Linear, Embedding, CrossEntropyLoss 
from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader 

from extract_training_data import FeatureExtractor

class DependencyDataset(Dataset):

  def __init__(self, input_filename, output_filename):
    self.inputs = np.load(input_filename)
    self.outputs = np.load(output_filename)

  def __len__(self): 
    return self.inputs.shape[0]

  def __getitem__(self, k): 
    return (self.inputs[k], self.outputs[k])


class DependencyModel(Module): # TODO Part 3
    def __init__(self, word_types, outputs):
        super(DependencyModel, self).__init__()
        self.embedding = Embedding(num_embeddings=word_types, embedding_dim=128)
        self.hidden = Linear(in_features=128*6, out_features=128)  # 6*128 since the embeddings were flattened
        self.output = Linear(in_features=128, out_features=outputs)  # 'outputs' - the size of the output layer

    def forward(self, inputs):
        # inputs shape: (batch_size, 6)
        embedded = self.embedding(inputs)  # Shape: (batch_size, 6, 128)
        embedded_flat = embedded.view(embedded.size(0), -1)  # Shape: (batch_size, 6*128)
        hidden_output = relu(self.hidden(embedded_flat))  # Shape: (batch_size, 128)
        output_logits = self.output(hidden_output)  # Shape: (batch_size, outputs)
        return output_logits



def train(model, loader): 

  loss_function = CrossEntropyLoss(reduction='mean')

  LEARNING_RATE = 0.01 
  optimizer = torch.optim.Adagrad(params=model.parameters(), lr=LEARNING_RATE)

  tr_loss = 0 
  tr_steps = 0

  # put model in training mode
  model.train()
 

  correct = 0 
  total =  0 
  for idx, batch in enumerate(loader):
 
    inputs, targets = batch
 
    predictions = model(torch.LongTensor(inputs))

    loss = loss_function(predictions, targets)
    tr_loss += loss.item()

    #print("Batch loss: ", loss.item()) # Helpful for debugging, maybe 

    tr_steps += 1
    
    if idx % 1000==0:
      curr_avg_loss = tr_loss / tr_steps
      print(f"Current average loss: {curr_avg_loss}")

    # To compute training accuracy for this epoch 
    correct += sum(torch.argmax(predictions, dim=1) == torch.argmax(targets, dim=1))
    total += len(inputs)
      
    # Run the backward pass to update parameters 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  epoch_loss = tr_loss / tr_steps
  acc = correct / total
  print(f"Training loss epoch: {epoch_loss},   Accuracy: {acc}")


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


    model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))

    dataset = DependencyDataset(sys.argv[1], sys.argv[2])
    loader = DataLoader(dataset, batch_size = 16, shuffle = True)

    print("Done loading data")

    # Now train the model
    for i in range(5): 
      train(model, loader)


    torch.save(model.state_dict(), sys.argv[3]) 


    # Obtained Result

    # 1st Epoch - Training loss epoch: 0.5790692997953181,   Accuracy: 0.8242665529251099
    # 2nd Epoch - Training loss epoch: 0.41787614724405453,  Accuracy: 0.8689920902252197
    # 3rd Epoch - Training loss epoch: 0.36042418692310324,  Accuracy: 0.8863770365715027
    # 4th Epoch - Training loss epoch: 0.3286465346579959,   Accuracy: 0.8959815502166748
    # 5th Epoch - Training loss epoch: 0.3077483263158519,   Accuracy: 0.9023163318634033
