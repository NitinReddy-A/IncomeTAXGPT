from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import timeit
load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY','f1769a09-e6c7-49e3-89c4-c1ba874265dd')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV', 'asia-southeast1-gcp-free')
#or
#PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY','9013a54a-3a33-40df-a8b2-6a069cd271f5')
#PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV', 'gcp-starter')

#***Extract Data From the PDF File***
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents

extracted_data=load_pdf_file(data='data/')

#print(data) 


#***Split the Data into Text Chunks****
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=text_split(extracted_data)
print("Length of Text Chunks", len(text_chunks))

#***Download the Embeddings from Hugging Face***
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

start = timeit.default_timer()
embeddings = download_hugging_face_embeddings()

query_result = embeddings.embed_query("Hello world")
print("Length", len(query_result))


#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="income2"
# or index_name="income"

#Creating Embeddings for Each of The Text Chunks
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)