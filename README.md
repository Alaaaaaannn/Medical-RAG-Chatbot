### Medical-Focused RAG Charbot

This is one of the projects I have done to learn and implement end-to-end RAG systems, from development all the way to deployment. This is the stack I will be using for this project:

1. Python for the programming language and wide access to GenAI related libraries.
2. Langchain: For agents creation, orchestration and coordination
3. LLMs like the GPT or Grok (free one) for inference and retrieval, along with learning from the document embeddings.
4. Flask for a Python-based stable backend to host the app online
5. Pinecone for vector DB to store document embeddings.
6. AWS especially EC2 to host the project on cloud along with a CI/CD pipeline (GitHub actions).

# My understanding of RAG (before and while implementing this project) 

- Traditional LLMs like ChatGPT or Gemini and trained on certain data and the versions pertaining to these models (like GPT 4.0) are only aware of information upto the date of the data that was fed to them. So, asking any queries regarding newer occurances will result in the LLM providing fake results that are unreliable.
- RAG is a newer methodology designed to bridge this problem. As we all know, machines only understand numbers, not text. But our newer information is often in multiple forms like text, images, audio, video, etc. which are not directly understood by machines or LLMs.
- This is where RAG (Retrieval Augmented Generation) comes in. From the data we are given, we embed or transform it, in my case into a vector form (vectors are collections of numbers, that store semantic meaning of the original data). This is done using embedding techniques like EfficientNetLite, FAISS, etc. This is called the knowledge base. The LLMs use this for the information retrieval.
- These embedded vectors then need to be stored in a database designed for this purpose, which is called a vector database or vector DB. In my project, I will be using Pinecone that offers a generous free tier and works for my data. This vector DB acts as the data store from which the LLMs infer new data to provide answers.
- For building a chatbot, traditionally using ChatGPT would result in our query being divided into chunks of words, each then getting embedded into numerical vectors. This vector is then inferred from the vector DB using multiple techniques like semantic similariy search or keywords search, from which it can derive vectors from the database that have similar values to our query, (like K-nearest Neighbors for a good example). Then it returns those queries.
- RAG works better than traditional finetuning, as finetuning would require edit of the original LLM weights, which would first of all require a lot of data, a lot of processing and obviously a huge expense. This is done by big tech companies though.

# Architecture of this project

- First of all we have the data we need to extract. In my project I have the Gale Encyclopedia of Medicine, as a pdf under data/ directory. This is the book from which we will extract data.
- This data is then divided into pieces or chunks via chunking, chunks corresponds to tokens in LLMs, more the tokens, more the strain on the LLM. What also matters is the size of each token, a higher token size would require larger processing per token and slows down the system a lot. Thats why chunk size and chunk overlap are two parameters we need to consider before implementing any RAG project. A decent chunk size ensures desirable performance, prevention of context overflow and session degradation as seen in common LLM sessions.
- GPT 4.0 has a context window of 8000 tokens, so assuming each token consists of 1000 tokens, it can hold up to 8 chunks per session.
- After chunking, we do the embeddings, using Embedding models, there are free and paid ones. For this project, we will be using Huggingface's sentence transformer.
- After this, we get the vector embeddings, a numeric representation of the data as embeddings. These are then stored onto our Pinecone vector DB, which acts as the knowledge database. This is run on the cloud.
- Vector DBs provide better functionality for LLM retrievals than traditional relational databases, mainly via their ability to provide key operations like semantic similarity or keyword search, to derive data similar and relevant to our query.
- User interacts with the system like this, they ask a query like "What are the symptoms of flu?", which goes through the embedding process just like the original data, and then goes through the vector DB, that returns the rank result. A parameter, k along with this rank result and the original question is passed to the LLM, which provides all the relevant context required by the LLM. Thus it can give the result tailored to the data that we have and avoid hallucinatiosn or wrong information.
- So to summarize, the architecture follows: Input data->Docs extraction->Chunking->Embedding models (Sentence transformer)->Vector embeddings->Vector DB (Pinecone).
- Retrieval or querying follows this architecture: User query->Chunking->Embedding models->Vector embeddings-> Inference with Vector DB to get rank result-> Current rank result, k value and original sentence passed to the LLM-> LLM result decoded into text and displayed back to the user.

# How does deployment work?

- As mentioned earlier, we are deploying this on an Amazon EC2 instance along with GitHub actions CI/CD pipeline. 
- Every commit via Github is passed and verified Github actions, that runs checks and notices errors, after which it containerizes the application into a Docker image, that gets stored onto the ECR or related database connected to the EC2. 
- EC2 is then port mapped to the IP address for the backend that gets exposed for the frontend to use.

# Project Setup

How to run?

STEPS:

1. Clone this repository: (under Git Bash) git clone https://github.com/Alaaaaaannn/Medical-RAG-Chatbot.git

2. Create a virtual environment using conda after cloning and opening repo directory: conda create -n medical-rag python=3.10 -y
and then: conda activate medical-rag

3. Install the requirements from the txt file: pip install -r requirements.txt
