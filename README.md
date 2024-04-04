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

# PlayGround:

1. System: How your model is going to behave.`You are a helpful assistant.
2. User
3. Model
   1. Model
   2. Temperature:controls randomness, 0 mean determininstc & repetitive which mean less creativity
   3. Max Length: token length
   4. Stop Sequences
   5. Top P: adds diversity..toP~0 means the model samples from a narrower selection of works. This makes output less random and diverse since the most probable token will be selected. AT Top P=0.1, only token comprising of the top 10% probability mass are considered.
   6. Frequency penalty: repetition penalty.. helps avoid same words too often
   7. Presence Penalty: The OpenAI Presence Penalty setting is used to adjust how much presence of tokens in the source material will influence the output of the model.

############################################################################################################

# Function Calling :

         `function calling` connects LLM to external tools(API):.

###################################################################################################################

# limitations of OpenAI api

1. openai model is not free.
2. can access limited models.(only openai)
3. data is not latest.

# with langchain:

1. you can access diff any model by using diffeent AOI
2. access private data resources
3. access any third party API
4. Langchain has :
   1. chains
   2. Document loader
   3. Agents
   4. access third party LLM(HF,etc)
   5. it can retain memory
   6. Prompt template

# Langchain is just a wrapper around openaAI API.

`prompt->langchain->opeanAi API(can be another)-> LLM`

it can also connect to data sources
`prompt->langchain->google/wiki/datasources`

# LANGCHAIN: with langchain you can access any LLM

LangChain is a robust library designed to simplify interactions with various large language model (LLM) providers, including OpenAI, Cohere, Bloom, Huggingface, and others. What sets LangChain apart is its unique feature: the ability to create Chains, and logical connections that help in bridging one or multiple LLMs.

Central to LangChain is a vital component known as LangChain Chains, forming the core connection among one or several large language models (LLMs).

` Prompt template + LLM -> LLMChain`
In certain sophisticated applications, it becomes necessary to chain LLMs together, either with each other or with other elements. These Chains empower us to integrate numerous components, weaving them into a cohesive application. Let’s delve deeper into the distinct types of Chains.

Through the creation of chains, multiple elements can seamlessly come together. Imagine this scenario: a chain is crafted to take in user input, polish it using a PromptTemplate, and subsequently pass on this refined response to a large language model (LLM). This streamlined process not only simplifies but also enriches the overall functionality of the system. In essence, chains serve as the linchpin, seamlessly connecting different parts of the application and enhancing its capabilities. Let’s summarize this:

Integrating prompt templates with LLMs allows for a powerful synergy.
By taking the output of one LLM and using it as input for the next, it becomes feasible to connect multiple LLMs in a sequential fashion.
Blending LLMs with external data enables the system to respond to inquiries effectively.
Integrating LLMs with long-term memory, such as chat history, enhances the overall context and depth of interactions.

# React

The Zero-shot ReAct Agent is a language generation model that can create realistic contexts even without being trained on specific data.

# Chains refer to sequences of calls - whether to an LLM, a tool, or a data preprocessing step.

# LANGCHAIN

1. Agents
2. Prompt templates
3. Chains
4. Document loader
5. Memory:

# Memory in Langchain

refers to remembering happening conversation.... retaining the memory like chatgpt.

A memory system needs to support two basic actions: reading and writing. Recall that every chain defines some core execution logic that expects certain inputs. Some of these inputs come directly from the user, but some of these inputs can come from memory. A chain will interact with its memory system twice in a given run.

1. AFTER receiving the initial user inputs but BEFORE executing the core logic, a chain will READ from its memory system and augment the user inputs.
2. AFTER executing the core logic but BEFORE returning the answer, a chain will WRITE the inputs and outputs of the current run to memory, so that they can be referred to in future runs.

# SETUP.py: # to install local pakage in virtual environment

# I want to consider `mcqgenerator` as a local python package.

Package is nothing but a folder containing multiple python files.
To treat folder as pkg, it must contain `__init__.py`
here `src` is a pkg

#######################################################################

# VECTOR DATABASE

is a database for storing high dimensional vector such as word embeddings and image embeddings.
A vector database stores pieces of information as vectors. Vector databases cluster related items together, enabling similarity searches and the construction of powerful AI models.

How do vector databases work?
Each vector in a vector database corresponds to an object or item, whether that is a word, an image, a video, a movie, a document, or any other piece of data. These vectors are likely to be lengthy and complex, expressing the location of each object along dozens or even hundreds of dimensions.

For example, a vector database of movies may locate movies along dimensions like running time, genre, year released, parental guidance rating, number of actors in common, number of viewers in common, and so on. If these vectors are created accurately, then similar movies are likely to end up clustered together in the vector database.

How are vector databases used?
Similarity and semantic searches: Vector databases allow applications to connect pertinent items together. Vectors that are clustered together are similar and likely relevant to each other. This can help users search for relevant information (e.g. an image search), but it also helps applications:
Recommend similar products
Suggest songs, movies, or shows
Suggest images or video
Machine learning and deep learning: The ability to connect relevant items of information makes it possible to construct machine learning (and deep learning) models that can do complex cognitive tasks.
Large language models (LLMs) and generative AI: LLMs, like that on which ChatGPT and Bard are built, rely on the contextual analysis of text made possible by vector databases. By associating words, sentences, and ideas with each other, LLMs can understand natural human language and even generate text.
To summarize: Vector databases work at scale, work quickly, and are more cost-effective than querying machine learning models without them.

# Embedding generation

## non dl (frequency based)

1. BOW(docmat)
2. TF-IDF
3. n-gram
4. One hot encoding
5. integer encoding

## issues with non-dl

### for One hot encoding & integer encoding

1. sparse matrix(too many zeroes)
2. no context

### for BOW(docmat), TF-IDF & n-gram

1. we create encoding using vocabularly
2. still no context
3. frequency based

## with dl

1. word2vec
2. fast text
3. ELMO
4. BERT
5. Glove(matrix factorization)

### benefits

1. creating dense vector
2. context-full

## WORD2VEC

# `based on features i.e. king has features`

we pass features into NN and we get embedding vector

Vector databases store embeddings. it indexes and store embeddings for faster retrieval and similarity search.

1. are used in searching
2. clustering where text strings are grouped by similarity
3. Recommendation: related items are recommended
4. classification

# Embeddings using openAI
