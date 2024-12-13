### The recurrent architecture
- Just like vectors, RNN show up in a lot of text-based apps as well.
- - But they have also been used together with CNNs for image-related purposes and in other sequence-driven domains.
- The basic idea behind RNNs is to make use of sequential information
- - In a traditional NN we assume that all inputs are independent of each other 
- - However, if you want to predict next word in a sentence, it makes sense to know words that came before it
- - RNNs are called recurrent because they perform the same task for every element of sequence, with the output being dependent on previous computations.
- RNNs are useful as they can handle variable length sequence inputs, in a way so they can deal with long-term dependencies, keep a memory of the imputs so far, building up an 
  internal 'stateful' representation of the input sequence.
- The instances are delivered one-by-one (or, think of every instance being a sequence)
- RNNs have a memory which captures information about what has been calculated so far
- Like human reasoning: humans don't start thinking from scratch every second, they remember what they've been thinking about
- Two popular types:
- - LSTM (Long Short-Term Memory)
- - GRU (Gated Recurrent Unit)

### ☞ Text classification with an RNN
-  see notebook "dle_rnn_textclassification"

### ☞ Text generation with an RNN
- see notebook "dle_rnn_textgeneration"

### Variants
- RNNs are frequently combined with CNNs for more advanced image tasks
- - R-CNN, Fast R-CNN, Faster R-CNN, YOLO, etc.
- CNN used to propose regions of interest, RNN used to classify them
- RNNs have also shown great success in many NLP tasks
- - Language modeling and generating text
- - Machine translation
- - Question-answering 
- - Chatbots
- - Speech recognition
- - Generating image descriptions
- Bidirectional RNN: similar goal, though the arhitecture setup is different
- Sequence-to-sequence model: consists of a encoder and decoder part
- - The encoder encodes a source to a context vector
- - The decoder takes in a context vector and generates a target
- - Used in e.g. translation tasks

### Attention and memory
- RNNs are the workhorse for any setting where one needs to work with sequences
- - Text, time series, audio...
- - Often combined with embeddings
- - And with CNNs as well 
- LSTMs have existed for quite some time
- Newer developments to focus on "attention" and "memory"
- Attention is loosely based on the visual attention mechanism found in humans
- Attention mechanisms have made their way into RNN architectures that are typically used in NLP and vision
- It seems somewhat unreasonable to have a model that can only focus on one part of the input at a time
- With an attention mechanism we no longer try encode full source sentence into a fixed-length state vector
- Rather, we allow network to attend to different parts of the source sequence at each step of output generation
- The basic problem that the attention mechanism solves is that it allows the network to refer back to the input sequnce, instead of forcing it to encode all information into 
  one fixed-length vector

### ☞ Text classification with attention
- See notebook "dle_rnn_textclassificationattention"

### ☞ Time series forecasting with an LSTM
- See notebook "dle_rnn_timeseries"

### Revisiting the CNN
- Instead of using recurrent approaches, (one-dimensional) CNNs can be used for sequence data as well
- - RNNs work great for text but convolutions can do it faster
- - Any part of a sentence can influence the semantics of a word. For that reason we want our network to see the entire input at once.
- - Problem of too many parameters solved with one dimensional convolutions
- - Just as for images, deconvolution can be  used to generate text.

### ☞ Text classification with a CNN
- See notebook "dle_cnn_textclassification"
