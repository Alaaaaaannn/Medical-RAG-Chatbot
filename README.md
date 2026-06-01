# Medical-Focused RAG Chatbot

The live website: https://alanfernandes-medical-rag-chatbot.hf.space

This is one of the projects I have done to learn and implement end-to-end RAG systems, from development all the way to deployment. This is the stack I will be using for this project:

1. Python for the programming language and wide access to GenAI related libraries.
2. Langchain: For agents creation, orchestration and coordination
3. LLMs like the GPT or Grok (free one) for inference and retrieval, along with learning from the document embeddings.
4. Flask for a Python-based stable backend to host the app online
5. Pinecone for vector DB to store document embeddings.
6. HuggingFace Spaces to host the whole Flask App (good free tier).

### My understanding of RAG (before and while implementing this project)

- Traditional LLMs like ChatGPT or Gemini and trained on certain data and the versions pertaining to these models (like GPT 4.0) are only aware of information upto the date of the data that was fed to them. So, asking any queries regarding newer occurances will result in the LLM providing fake results that are unreliable.
- RAG is a newer methodology designed to bridge this problem. As we all know, machines only understand numbers, not text. But our newer information is often in multiple forms like text, images, audio, video, etc. which are not directly understood by machines or LLMs.
- This is where RAG (Retrieval Augmented Generation) comes in. From the data we are given, we embed or transform it, in my case into a vector form (vectors are collections of numbers, that store semantic meaning of the original data). This is done using embedding techniques like EfficientNetLite, FAISS, etc. This is called the knowledge base. The LLMs use this for the information retrieval.
- These embedded vectors then need to be stored in a database designed for this purpose, which is called a vector database or vector DB. In my project, I will be using Pinecone that offers a generous free tier and works for my data. This vector DB acts as the data store from which the LLMs infer new data to provide answers.
- For building a chatbot, traditionally using ChatGPT would result in our query being divided into chunks of words, each then getting embedded into numerical vectors. This vector is then inferred from the vector DB using multiple techniques like semantic similariy search or keywords search, from which it can derive vectors from the database that have similar values to our query, (like K-nearest Neighbors for a good example). Then it returns those queries.
- RAG works better than traditional finetuning, as finetuning would require edit of the original LLM weights, which would first of all require a lot of data, a lot of processing and obviously a huge expense. This is done by big tech companies though.

### Architecture of this project

- First of all we have the data we need to extract. In my project I have the Gale Encyclopedia of Medicine, as a pdf under data/ directory. This is the book from which we will extract data.
- This data is then divided into pieces or chunks via chunking, chunks corresponds to tokens in LLMs, more the tokens, more the strain on the LLM. What also matters is the size of each token, a higher token size would require larger processing per token and slows down the system a lot. Thats why chunk size and chunk overlap are two parameters we need to consider before implementing any RAG project. A decent chunk size ensures desirable performance, prevention of context overflow and session degradation as seen in common LLM sessions.
- GPT 4.0 has a context window of 8000 tokens, so assuming each token consists of 1000 tokens, it can hold up to 8 chunks per session.
- After chunking, we do the embeddings, using Embedding models, there are free and paid ones. For this project, we will be using Huggingface's sentence transformer.
- After this, we get the vector embeddings, a numeric representation of the data as embeddings. These are then stored onto our Pinecone vector DB, which acts as the knowledge database. This is run on the cloud.
- Vector DBs provide better functionality for LLM retrievals than traditional relational databases, mainly via their ability to provide key operations like semantic similarity or keyword search, to derive data similar and relevant to our query.
- User interacts with the system like this, they ask a query like "What are the symptoms of flu?", which goes through the embedding process just like the original data, and then goes through the vector DB, that returns the rank result. A parameter, k along with this rank result and the original question is passed to the LLM, which provides all the relevant context required by the LLM. Thus it can give the result tailored to the data that we have and avoid hallucinatiosn or wrong information.
- So to summarize, the architecture follows: Input data->Docs extraction->Chunking->Embedding models (Sentence transformer)->Vector embeddings->Vector DB (Pinecone).
- Retrieval or querying follows this architecture: User query->Chunking->Embedding models->Vector embeddings-> Inference with Vector DB to get rank result-> Current rank result, k value and original sentence passed to the LLM-> LLM result decoded into text and displayed back to the user.

### Project Setup

How to run?

STEPS:

1. Clone this repository: (under Git Bash) git clone https://github.com/Alaaaaaannn/Medical-RAG-Chatbot.git

2. Create a virtual environment using conda after cloning and opening repo directory: conda create -n medical-rag python=3.10 -y
   and then: conda activate medical-rag

3. Install the requirements from the txt file: pip install -r requirements.txt

4. Add your secret keys: make a copy of `.env.example` and rename it to `.env`, then paste in your Pinecone and Groq API keys (these are free to get from their websites).

5. Build the knowledge base (do this once): run `python store_index.py`. This reads the medical PDF, breaks it into small pieces, turns them into numbers (embeddings), and uploads them to Pinecone. It also saves a `chunks.pkl` file the app uses for searching.

6. Start the chatbot: run `python app.py` and open `http://localhost:8080` in your browser.

---

## Everything this project can do

I started with a basic idea and kept adding features. Here is what each one does, in plain English.

### The chat website (the part you interact with)

A clean, ChatGPT-style chat page (built with plain HTML, CSS and JavaScript):

- A welcome screen with example questions you can click.
- A typing animation while the bot "thinks".
- Works on phones and computers.

### Smarter answers (making the AI search better)

These features help the bot find the _right_ information from the medical book and answer accurately:

- Better book pieces (chunking): the book is split into reasonably sized pieces so each one keeps its full meaning, and blank/junk pieces are thrown away.
- Smarter understanding (embeddings): I upgraded the model that turns text into numbers to a better one (called BGE), so searches find more relevant pages.
- Two ways to search at once (hybrid search): it searches both by exact keywords (good for medical terms) and by meaning, then combines them.
- Double-checking the results (reranking): a second AI model re-sorts the search results and keeps only the few best ones before answering.
- Asking the question in different ways (multi-query): the AI rephrases your question a few times so it doesn't miss good results just because of how you worded it.
- Remembers the conversation (memory): you can ask follow-up questions like "what causes it?" and it knows what "it" means.
- Shows where the answer came from (citations): every answer lists the source page numbers, so you can trust and verify it.
- Honesty check (grounding): if the answer isn't actually supported by the medical book, the bot says it doesn't know instead of making something up.

### Nicer to use

- Nicely formatted replies (markdown): lists, bold text and tables show up properly.

### Safe and hard to misuse (protecting my free Groq quota)

Since the AI runs on a free plan, I added protections so it can't be abused or run up costs:

- Emergency safety: if someone types something urgent (like "chest pain" or "can't breathe"), the bot stops and tells them to call emergency services instead of giving advice.
- Answer length limit: the AI can't write a huge reply, so no one can trick it into wasting tokens.
- Daily usage limit: there's a cap on how many questions the whole app answers per day; after that it politely says "try again tomorrow".
- Message length limit: very long messages are rejected before they reach the AI.
- Memory cleanup: old conversations are automatically forgotten so the app doesn't run out of memory.
- Secret key: the app uses a proper secret so user sessions can't be faked.

### Easy settings (no code editing needed)

All the important settings (which AI model to use, how long answers can be, the daily limit, etc.) live in the `.env` file. You can change them without touching any code. See `.env.example` for the full list.

### Where things live (project files)

- `app.py` — the web server that runs the chatbot.
- `store_index.py` — the one-time script that builds the knowledge base.
- `src/` — the building blocks: `config.py` (settings), `helper.py` (reading/splitting the PDF + embeddings), `retriever.py` (searching), `chain.py` (asking the AI), `prompter.py` (the bot's instructions and safety rules), `cache.py` (daily usage limit + conversation memory).
- `templates/` and `static/` — the chat website (HTML, CSS, JavaScript).
- `research/enhanced_rag.ipynb` — a notebook to test all the features step by step.
- `FEATURES.md` — a more technical write-up of all the features.
