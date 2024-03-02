# GenAI

# Genereative AI

1. Generative Image Model (GANs)
2. Generative Language Model:
   _. can also generate images
   _, Image to Image is also possible in (2)
   _. input->input prompt
   _. output->output prompt

# sequence to sequence mapping techniques:

1. one to one mapping
2. one to many
3. many to many mapping
4. many to one mapping

# a bit conext

to process sequence data( sequence to sequence mapping)

1. RNN: feedback loop: pass output to hidden layer; short term memory.
2. LSTM: cell state; deals with long sentences; along with timestamps(hidden layers), there is a cell state;3 gates:
   _._.Forget,
   _._.input,
   \_.\*. output
3. GRU:updated LSTM verision; no cell state; everythong is done by hidden state itself; 2 gates:
   _, reset
   _.update gate

## after 2019
