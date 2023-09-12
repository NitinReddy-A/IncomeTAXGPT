import streamlit as st
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from dotenv import load_dotenv
import os
import timeit
import sys
load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY','f1769a09-e6c7-49e3-89c4-c1ba874265dd')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV', 'asia-southeast1-gcp-free')
#or for index "income"
#PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY','9013a54a-3a33-40df-a8b2-6a069cd271f5')
#PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV', 'gcp-starter')

#***Download the Embeddings from Hugging Face***
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

start = timeit.default_timer()
embeddings = download_hugging_face_embeddings()

#query_result = embeddings.embed_query("Hello world")
#print("Length", len(query_result))


#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="income2"
# or index_name="income"

#Creating Embeddings for Each of The Text Chunks
#docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)




#If we already have an index we can load it like this
docsearch=Pinecone.from_existing_index(index_name, embeddings)

query = "What is income tax"

#docs=docsearch.similarity_search(query, k=3)

#print("Result", docs)

prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})
#llm=ChatOpenAI(model_name="gpt-3.5-turbo")

qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={'k': 2}),return_source_documents=True, chain_type_kwargs=chain_type_kwargs)

#query="What are Allergies"

#print("Response",qa.run(query))


st.title("IndianTaxGPT :books:")
user_input = st.text_input("Enter your question:")
if st.button("Generate"):
    with st.spinner("Processing"):
        if user_input:
            result = qa({"query": user_input})
            print(type(user_input),"---",user_input)
            st.write("Response:", str(result["result"]))

# Add an exit button
if st.button("Exit"):
    st.write("Exiting")
    sys.exit()



end=timeit.default_timer()

print(f"Time to retrieve response: {end-start}")