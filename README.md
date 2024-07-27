# IndianTaxGPT

IndianTaxGPT is an innovative project aimed at addressing the lack of accessibility to accurate and understandable information about Indian tax laws and filing procedures. Our platform serves as a virtual tax consultant, leveraging AI-powered technologies to provide individuals and enterprises with reliable, easy-to-understand information about Indian tax regulations.

## Problem Statement

The problem we’re addressing is the lack of accessibility to accurate and understandable information about Indian tax laws and filing procedures. Many individuals and budding enterprises struggle with navigating complex tax regulations, leading to errors, missed deductions, and non-compliance. Limited access to affordable tax professionals further compounds this challenge, hindering informed financial decisions.

## Solution

IndianTaxGPT aims to revolutionize the way the common man in India understands taxes and manages their tax-related affairs. Our platform utilizes advanced technologies such as Natural Language Processing (NLP) and GPT-powered models to offer accurate, reliable, and easy-to-understand information about Indian tax laws, filing procedures, deductions, exemptions, and more. By providing a user-friendly and accessible interface, we empower users to make informed financial decisions and ensure compliance with tax regulations.

## Challenges Faced

1. **Cost Limitations**: We encountered challenges due to limited resources, particularly in accessing free alternatives for vector databases and utilizing OpenAI models. As a result, our prototype focuses solely on the income tax category within the subset of taxes.

2. **Performance**: To mitigate cost limitations, we opted for open-source LLM and embeddings models instead of OpenAI models. However, this decision led to slower performance compared to OpenAI models.

3. **Deployment Constraints**: Utilizing open-source Langchain models required downloading them locally and converting embeddings with CPU computations, making it difficult to generate a public link for showcasing. However, our prototype can be tested using the code available on GitHub, along with the downloaded LLM model.

## Technologies Used

- **Natural Language Processing (NLP)**: Leveraging NLP techniques to process and understand textual data related to Indian tax laws and procedures.
- **GPT-powered Models**: Utilizing GPT-powered models to provide accurate and comprehensive information to users.[Llama 2-7b V]
- **Embeddings Vector Search Database**: Implementing embeddings vector search databases to enable efficient data retrieval and processing.[Huggingface Embeddings Transformer and Pinecone Vector DB]
- **Langchain (LLM)**: Incorporating Langchain interface various components.
- **Deployment**: Developing a user-friendly web interface using Streamlit.

## Technical Overview

This project utilizes Pinecone VectorDB to store taxation data in the form of embeddings. The Large Language Model (LLM), specifically llama-2, is integrated with VectorDB to handle the logic of similarity search. It compares the embeddings present in the database with the embeddings obtained from converting the user's question into the required format.

## Getting Started

1. Download all dependencies listed in `requirements.txt` and `models/requirements.txt`.
2. Place all relevant data in PDF format into the `data` folder.
3. Create a Pinecone account and set up an index.
4. Run `embedder.py` with your own API keys and index name to embed the taxation data.
5. Run `app.py` with your own API keys and index name to start the application.

## Screenshots

![image](https://github.com/NitinReddy-dev/IncomeTAXGPT/assets/114919978/a0666954-e1e2-4949-badd-217f3058db0f)

![image](https://github.com/NitinReddy-dev/IncomeTAXGPT/assets/114919978/867cc00d-a045-4de5-9736-a81ee5c4d4de)

We appreciate your interest in IncomeTaxGPT and welcome any feedback or contributions to improve the platform.
