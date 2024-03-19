# IncomeTAXGPT
This project is part of a hackathon called Start-a-thon and is meant to be a ChatGPT like tool specifically for Indian tax.

## Technicalities:
This project makes use of Pinecone VectorDB to store the taxation data in the form of embeddings. The LLM (llama-2) is then connected with the VectorDB which handlles the logic of similarity search of the embeddings present in the DB with the embeddings obtained from converting the posted question to required format.

How to run:

1. Download all dependencies listed in requirements.txt and models/requirements.txt

2. Place all relevant data in pdf format into the data folder

3. Create a Pinecone account and create an index

4. Run embedder.py with your own API keys and index name

5. Run app.py with your own API keys and index name
