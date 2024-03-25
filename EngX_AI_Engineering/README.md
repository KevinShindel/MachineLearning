# EngX AI-Supported Engineering

### Module 1: Mastering LLMs
- In the first module, you will explore Language Learning Models (LLMs), gaining insights into their potential to enhance software development processes. You will learn how to effectively communicate with conversational and inline tools, like ChatGPT and GitHub Copilot, through crafting prompts and providing context.

### Module 2: Coding
- In the second module, you will delve into the practical applications of LLMs in coding and explore their role in accelerating new feature creation, streamlining maintenance, and improving refactoring processes. By the end of this module, you will possess the knowledge to leverage tools like ChatGPT and GitHub Copilot effectively, ultimately boosting your software development efficiency.

### Module 3: Development Testing
- In the third module, you will explore how LLMs can enhance development testing, including new test creation and test data generation. By the end of this module, you will be equipped to leverage LLM-driven testing strategies for more efficient and reliable software development practices.

### Module 4: Technical Documentation
- In the fourth module, you will explore the role of LLMs in streamlining technical documentation creation and maintenance. By the end of this module, you will be equipped to leverage LLMs like ChatGPT and GitHub Copilot to generate high-quality documentation that fosters improved collaboration and project success.

## Practical Tasks
- Throughout EngX AI-Supported Engineering, you will have the opportunity to practice what you have learned from lessons. You will engage in tasks that will help you identify possible areas for growth in communication with AI tools skills.
- Pay attention that only tasks after lesson "Creating a New Feature" are mandatory. All the rest of practices are optional and may be skipped.
- All tasks can be completed in one of four programming languages: Java, JavaScript/TypeScript, C#, and Python. Please limit your focus to your primary language; you are not required to complete tasks in multiple languages.


## Intro: 

- **Machine Learning (ML)**

Machine learning is a branch of AI that focuses on detecting patterns and making predictions using algorithms and statistical techniques.

- **Deep Learning (DL)**

Deep learning is a subset of ML that uses multi-layered artificial neural networks to deliver state-of-the-art accuracy in object detection, speech recognition, and language translation tasks. It imitates the mechanism of the human brain to interpret data, such as images, sound, and text.

- **Natural Language Processing (NLP)**

Natural language processing is a subset of AI and based on ML, extended with language algorithms. NLP covers the interaction between computers and human languages. NLP enables machines to understand, generate, and manipulate natural language texts and speech. At a high level, NLP takes human language as an input, processes it, and produces an output, like an action or a response.

- **Large Language Model (LLM)**

A large language model consists of a neural network with many parameters trained on large quantities of text. LLMs are a type of ML that uses large amounts of data to develop models for understanding text. LLMs process natural language inputs and predict the next word based on what they have already seen. LLMs can also generate new texts based on a given prompt or context.


### Conversational AI tools
- Conversational GenAI tools are designed to simulate human-like conversation. They understand the context, respond to prompts, and can generate human-like text, making them useful for tasks such as drafting emails, writing articles, or powering chatbots.
- Examples of conversational AI tools include ChatGPT, GPT-3, and OpenAI's Codex.

### Inline AI tools
- Inline GenAI tools are integrated directly into IDE and act while you're writing the code, assisting you with coding tasks. They analyze the context of the code and provide real-time suggestions or corrections, akin to having an expert coder guiding you. They can predict and generate code snippets, making them valuable tools for software development.
- Examples of inline AI tools include GitHub Copilot, TabNine, and Kite.

### Discriminative AI
- While generative models learn about the distribution of the dataset, discriminative models learn about the boundary between classes within a dataset. With discriminative models, the goal is to identify the decision boundary between classes to apply reliable class labels to data instances. Discriminative models separate the classes in the dataset by using conditional probability, not making any assumptions about individual data points.
- Examples of discriminative models in machine learning include support vector machines, logistic regression, decision trees, and random forests.

### Generative AI
- Generative models learn the distribution of the dataset and generate new data points that resemble the training data. These models can create new samples that are similar to the training data, making them useful for tasks like image generation, text generation, and data augmentation.
- Examples of generative models in machine learning include generative adversarial networks (GANs), variational autoencoders (VAEs), and autoregressive models.


### Transformer LLM Architecture
- Transformer is a type of model architecture that processes a text all at once rather than one word at a time and has a strong ability to understand the relationship between those words.
- The number and the size of layers in LLM architecture vary based on the model.


#### Transformers contains: 
 - Input - You provide an input, also called a prompt.
 - Embedding - LLMs are mathematical functions whose input and output are lists of numbers. Consequently, words must be converted to numbers. The process starts with LLM breaking input text down into tokens, the basic units of meaning in a language. Tokens can be words, phrases, or even punctuation marks. Then, by employing neural networks, all the tokens are mapped to numerical representations in a process known as embedding. These representations are used by the transformer model for further processing.
 - Positional Encoding - The model processes the numerical inputs derived from text using many layers of neural networks. An essential part of this processing is positional encoding, a feature that helps keep track of the tokens' order within a sentence. After tokens are transformed into numerical formats, positional encoding assigns an extra value to each one, indicating its position in the sentence. This is crucial because swapping words, as in 'The cat chased the dog' and 'The dog chased the cat' sentences, changes the meaning entirely. So, by adding positional values, the model can correctly interpret the order and meaning of each word and word sequence.
 - Self-Attention Mechanism - Self-attention connects tokens as they progress through the model's levels. The self-attention mechanism assists the model in recognizing the context and mutual dependencies of all words in the input text. Importantly, it considers the entire input sequence when creating each token in the output. This understanding is crucial for the model to assess the semantic importance of each word and make accurate predictions during the output generation process.
 - Decoder - The decoder, similar to the encoder, is constructed of multiple layers of neural networks. These layers also incorporate positional encoding and self-attention mechanisms. The purpose of the decoder is to generate a set of possible subsequent words (tokens) based on the encoded inputs. The output of the decoder (and thus, the LLMs as a whole) is a probability distribution over its vocabulary. In other words, the decoder assigns probabilities to all possible words in its vocabulary for each generated token.
 - Output - Once the input prompt is processed, the model generates a response one token at a time. Typically, the word with the highest probability is selected as the next token in the response. However, another approach can be used, where the next token is sampled from the distribution of the predicted probabilities. This adds some variability and creativity to the model's output.

## Hallucinations
- Refers to generating content that appears semantically or syntactically plausible but is factually incorrect or unrelated to the provided context.

Hallucinations can happen for a variety of reasons, including:

+ There is an error in encoding and decoding between text representations.
+ The training dataset contains incorrect or nonsensical information.
+ There is insufficient context in user input to respond correctly or meaningfully.
+ The LLM is simply making things up.

Factual Inaccuracies - This type of hallucination includes generating text that contains factual errors or that is not based on reality due to a lack of explicit real-world knowledge.
- Why did it happen?
- GPT does not inherently know the facts of the world. It generates text based on the patterns it learned during training. It cannot verify data in real-time or access updated information and does not possess explicit knowledge about factual events. As a result, it sometimes produces factually incorrect outputs.

Fabrication or Misrepresentation - This type of hallucination includes creating artificial sources, making unfounded claims, or designing entirely fictional scenarios.

+ Why did it happen?
+ GPT learned to generate text based on patterns in the training data. There are multiple examples of URLs being shared in the context of referencing articles, sources, or additional information in this data. As a result, GPT recognized these patterns and learned to produce fake URLs in similar contexts. These generated URLs, however, are not based on genuine data but rather on learned patterns.

Nonsensical Output - Conversational AI tools may sometimes respond with completely random, unrelated, or nonsensical in the real-world context answers.

+ Why did it happen?
+ Nonsensical outputs happen primarily due to the nature of AI language models like GPT. They don't truly understand language in the way humans do. Furthermore, it cannot access real-world sensory experiences or common-sense knowledge to validate its responses against reality. The model generates text based on patterns it identifies in the data it was trained on. The model may produce nonsensical output if a particular pattern matches nonsensical data. This issue is further complicated by the model's occasional tendency to "improvise" when unsure about appropriate responses, leading to the potential generation of unlikely or illogical content. Additionally, because GPT-3 was trained on massive amounts of internet text, it incorporates various contextual errors, inconsistencies, and conflicting information from its training sources.

Biased Output - Due to the nature of its training, GPT may sometimes reproduce or magnify the biases present in the training data it was modeled on.

+ Why did it happen?
+ This happens because machine learning models like GPT-3 learn from their training data. If the training data contain biases—which can often be the case with large-scale internet text data—the model can learn and propagate these biases. In this case, the stereotype of female nurses is relatively common in text data, and hence GPT-3 may have learned to associate nurses with female pronouns.

### Validating Results

- Cross-checking the generated content against reliable sources or using fact-checking tools can help identify factual inaccuracies.
- Appropriateness of the generated content can be assessed by evaluating it against the context and prompt provided.
- Critical thinking and common sense can help in identifying nonsensical or biased content.


## Prompting
Prompting is giving instructions to an AI system to perform a task.


### Prompt Structure
- Instructions: Clearly define the task or context for the AI model.
- Context: Provide relevant information or background to help the model understand the task.
- Input data: Include any necessary data or examples required for the task.
- Output: Specify the expected output or the type of response you are looking for.

### AI Inline tools
- Copilot - GitHub Copilot, a result of collaboration between GitHub and OpenAI, is an advanced code completion tool powered by artificial intelligence (AI). Copilot is an AI assistant that suggests comprehensive lines or blocks of code as you type, similar to auto-complete, but more advanced.

Copilot architecture:
- Client
- Server-side model

```text
GitHub Copilot sends the code or comment you input to a server where the Codex model is running in real time. 
The server then processes the input and generates code suggestions based on its vast library of programming patterns and syntax.
 The model then swiftly generates a prediction for what should come next and sends it back to your editor in real time.
Copilot's main function involves turning comments into code. 
You create a comment that describes the logic you need, 
and Copilot will automatically generate suggestions.
```