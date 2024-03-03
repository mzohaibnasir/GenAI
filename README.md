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

## after 2019U an encoder/fedcoder. Transformers were introduced

# LLM:

is a trained DL model that understands and generate text in a human like fashion. LLMs are good at understanding and gnenerating human language. Large because of the size and complexity of NN and size of training dataset.

A single LLM can :
\*. Text classification
_. Text generation,
_. chatbot
_. summarizer
_. Translation
\*. Code generation, etc

# LLMs are based on transformer.

# few milestones in LLM:

_. BERT
_. GPT
_. XLM
_. T5
_. Megatron
_. M2M

# But some of LLMs are using encoder , some are using decoder and some are using both.

# API

\*. https://platform.openai.com/
\*. https://www.ai21.com/

# Transfer Learning vs Fine Tuning

Transfer learning involves using a pre-trained model as a starting point and freezing all the pre-trained layers while training only the new layers added on top. Fine-tuning, on the other hand, allows the pre-trained layers to be updated during training, which can lead to better performance on the new task.

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

# OpenAi model

1. Name: gpt-3.6-turbo-1106
2. Description:
3. context window: 16835 token as input, return 1096 tokens ::: it means 'our input/output prompts are nothing but collections of tokens. The context window in OpenAI models like GPT-3.5 Turbo refers to the maximum number of tokens or words that the model can consider at a time when generating responses. For example, the GPT-3.5 Turbo model with a context window of 16,385 tokens can support approximately 20 pages of text in a single request. This large context window allows the model to process and understand extensive amounts of information, enabling more accurate and contextually relevant responses. The context window size is crucial as it determines the amount of information the model can use to generate coherent and meaningful outputs
4. Training data:
