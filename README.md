# GenAI

# Genereative AI

1. Generative Image Model (GANs)
2. Generative Language Model:
   _. can also generate images
   _, Image to Image is also possible in (2)
   _. input->input prompt
   _. output->output prompt

# sequence to sequence mapping techniques: can be implemented by all RNN, LSTM, GRU

1. one to one mapping
2. one to many i.e. image capturing
3. many to many mapping : language translation
4. many to one mapping i.e. sentiment analysis

# a bit of context for LLM

to process sequence data( sequence to sequence mapping)

1. RNN: feedback loop: pass output to hidden layer; short term memory.
2. LSTM: cell state; deals with long sentences; along with timestamps(hidden layers), there is a cell state;3 gates:
   _._.Forget,
   _._.input,
   \_.\*. output
3. GRU:updated LSTM verision; no cell state; everythong is done by hidden state itself; 2 gates:
   _, reset
   _.update gate

## ISSUE: with this seq to seq mappigng is whatever output we are getting is depended on input.

if input is of 5 length , output would be 5 too. so fixed length input and output

## to cope with this of fixed dimensionality: encoder decoder(paper: sequence to sequence with nn )

Encoder(RNN, LSTM, GRU) -> context vector(latent) -> decoder(RNN, LSTM, GRU)

encoder decoder architecture was able to deal with fixed dimensonality issue but still wasn't able to sustain the context for longer sentences.

## to cope with that comes attention inside context:

1. https://cjlise.github.io/machine-learning/Neural-Machine-Translation-Attention/
2. http://jalammar.github.io/illustrated-transformer/
3. https://youtu.be/SMZQrJ_L1vo

## after 2019
