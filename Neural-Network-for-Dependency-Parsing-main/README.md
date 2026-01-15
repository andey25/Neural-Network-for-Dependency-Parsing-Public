# Neural Network for Transition-Based Dependency Parsing


This repository implements a **greedy, transition-based dependency parser** trained with a small neural network. It follows the pipeline: (1) read gold dependency trees in **CoNLL format**, (2) generate supervised training examples using an **oracle** over parser configurations (stack/buffer), (3) train a neural classifier to predict the next parser action (SHIFT / LEFT-ARC(label) / RIGHT-ARC(label)), and (4) decode new sentences by repeatedly selecting the highest-scoring **legal** transition until the buffer is empty. Evaluation reports **UAS/LAS** (unlabeled/labeled attachment scores).

At a high level, the model learns a policy for an **arc-standard** style parser using a fixed feature template: the top 3 items of the stack and the first 3 items of the buffer (6 token positions total), represented via learned embeddings.

---

## Repository layout

- `conll_reader.py`  
  Parses CoNLL dependency trees into a `DependencyStructure` of `DependencyEdge`s and provides helpers to print trees.

- `get_vocab.py`  
  Builds vocabularies from a CoNLL dataset:
  - `words.vocab`: word-to-id mapping with special tokens (`<CD>`, `<NNP>`, `<UNK>`, `<ROOT>`, `<NULL>`)
  - `pos.vocab`: POS-tag-to-id mapping

- `extract_training_data.py`  
  Core training-data generation:
  - Defines a parser `State` (stack/buffer/deps) with `shift`, `left_arc`, `right_arc`
  - Implements `FeatureExtractor` to produce:
    - input features: 6 vocabulary ids (top-3 stack + top-3 buffer positions)
    - output labels: one of `shift`, `left_arc(rel)`, `right_arc(rel)` for each dependency relation
  - Produces `input_*.npy` and `target_*.npy` matrices for training.

- `train_model.py`  
  Defines and trains `DependencyModel`:
  - Embedding layer (128d) for the 6 input ids
  - MLP: flatten (6×128) → hidden(128) → output(|labels|)
  - Trains using `CrossEntropyLoss` and `Adagrad`
  - Saves the trained model weights (e.g., `model.pt`)

- `decoder.py`  
  Loads the trained model and runs **greedy decoding**:
  - At each step, scores all transitions and applies the highest-probability legal action
  - Outputs a predicted dependency tree in CoNLL format.

- `evaluate.py`  
  Compares predicted vs gold trees and reports:
  - Micro-averaged and macro-averaged **LAS** and **UAS**

---

## Setup

### Environment
This project uses **Python 3**, plus:
- `numpy`
- `torch`

If you want a minimal install:
```bash
pip install numpy torch
```

---

## Typical workflow

### 1) Build vocabularies from a CoNLL file
```bash
python get_vocab.py data/train.conll data/words.vocab data/pos.vocab
```

### 2) Extract training matrices (inputs/targets)
```bash
python extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy
```

### 3) Train the parser model
```bash
python train_model.py data/input_train.npy data/target_train.npy model.pt
```

### 4) Decode / parse a CoNLL file
```bash
python decoder.py model.pt data/dev.conll > dev.pred.conll
```

### 5) Evaluate UAS/LAS against gold
```bash
python evaluate.py model.pt data/dev.conll
```

---

## Notes / implementation details

- **Features (6 positions):** the feature extractor uses a fixed template consisting of:
  - top-3 items from the stack, and
  - top-3 items from the buffer  
  Missing positions are filled with `<NULL>`. The root token is represented as `<ROOT>`.

- **Word normalization:** tokens with POS `CD` and `NNP` are mapped to `<CD>` and `<NNP>` rather than treated as normal words.

- **Output space:** the classifier predicts among:
  - `shift`
  - `left_arc(relation)` for every dependency relation seen in training
  - `right_arc(relation)` for every dependency relation seen in training

- **Decoding is greedy:** `decoder.py` does not run beam search; it simply takes the top-scoring legal transition each step.

---

## Reproducing the provided results
The repo contains example “Obtained Result” comments in `train_model.py` and `evaluate.py` showing typical training loss/accuracy and UAS/LAS numbers for a particular dataset split and hyperparameters.
