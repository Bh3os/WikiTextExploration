| Name                         | University ID |
| ---------------------------- | ------------: |
| Elbaraa Hatem Fathy Ismaeil  |      22101464 |
| Amar Hassan Ali              |      22100179 |
| Farah Mohamed Hassan Shehata |      22101433 |


# Language Model Report

## Introduction

This README file summarizes the implementation and evaluation of N-gram and RNN-based language models on the WikiText dataset. We explore different model architectures, training processes, and address challenges associated with language modeling.

## Approach

### Data Preprocessing

The data preprocessing steps involve:

*   **Tokenization:** Using `nltk.tokenize` to split the text into sentences and words.
*   **Lowercasing:** Converting all words to lowercase.
*   **Vocabulary Building:** Creating a vocabulary from the training data and handling out-of-vocabulary words with an `<oov>` token.
*   **Start and End Tokens:** Adding `<s>` and `</s>` tokens to the beginning and end of each sentence.
*   **Cleaning Tokens:** Removing punctuation and handling special tokens like `@-@` and `@.@`.

The `Prep` class in `data_prep.py` and `rnn_model.ipynb` implements these steps. The `Vocab` class in `data_module.py` converts word tokens to indices and vice versa, creating data and target lists based on a given window size.

### Model Architectures

We implemented the following model architectures:

*   **N-gram Models:** Bigram and trigram models with Laplace smoothing to handle unseen n-grams. The `bigram_model_.ipynb` notebook implements these models.
*   **RNN Model:** A basic RNN model with an embedding layer, an RNN layer with 2 layers and a hidden size of 100, and a fully connected layer for output. The `rnn_pl.py` and `rnn_model.ipynb` files implement this model.
*   **LSTM Model:** An LSTM model with an embedding layer, an LSTM layer with 2 layers and a hidden size of 100, and a fully connected layer for output. The `lstm_pl.py` and `rnn_model.ipynb` files implement this model.

The RNN and LSTM models are implemented using PyTorch Lightning for training and logging.

### Training Process

The RNN and LSTM models are trained using the following process:

*   **Optimizer:** SGD with a learning rate of 0.5 for RNN and 5 for LSTM.
*   **Loss Function:** Cross-entropy loss.
*   **Batch Size:** 20.
*   **Number of Epochs:** 20.
*   **Logging:** Using TensorBoard to log training and validation loss and perplexity.

The `TextLightningModule` (in `rnn_pl.py`) and `TextLSTMModule` (in `lstm_pl.py`) classes define the training steps.

## Experimental Results

### Perplexity

The following table shows the perplexity results for the different models:

| Model          | Perplexity |
| -------------- | ---------- |
| Bigram         | 3621.65    |
| Trigram        | 36298.54   |
| RNN  | 102.39926147460938   |
| LSTM  | 76.4348373413086    |

Note: The RNN and LSTM perplexity values are not available in the provided files.

### Visualizations

TensorBoard was used to visualize the training and validation loss and perplexity for the RNN and LSTM models. However, the visualizations are not included in this report.

## Challenges Faced

*   **Gradient Issues:** RNNs are known to suffer from vanishing and exploding gradients, which can make training difficult. LSTM models are designed to mitigate these issues.
*   **N-gram Sparsity:** N-gram models suffer from sparsity, especially for higher-order n-grams. Laplace smoothing is used to address this issue.
*   **Handling Large Vocabularies:** Large vocabularies can increase the computational cost of training and evaluating language models.
*   **Computational Constraints:** Training large language models requires significant computational resources.

## Insights

*   N-gram models are simple to implement but suffer from sparsity and cannot capture long-term dependencies.
*   RNNs can capture long-term dependencies but are more difficult to train due to gradient issues.
*   LSTMs are a variant of RNNs that are designed to mitigate gradient issues and can achieve better performance.

## Error Analysis

Error analysis was not performed due to the limited information available in the provided files. However, potential sources of error include:

*   **Data Noise:** The WikiText dataset may contain noise, which can affect the performance of the models.
*   **Sparsity:** Even with Laplace smoothing, N-gram models may suffer from sparsity, especially for rare n-grams.
*   **Short vs. Long-Term Dependencies:** RNNs and LSTMs may struggle to capture very long-term dependencies.
