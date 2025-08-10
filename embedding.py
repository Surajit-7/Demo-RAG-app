from loadDocs import load_docs
from dotenv import load_dotenv
# from langchain_community.embeddings import OpenAIEmbeddings
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

documents = load_docs()

# openaiKey = os.getenv("OPENAI-KEY") -> to test if openai api key is imported or not

# print(openaiKey)
def do_embedding():

    embedding = OpenAIEmbeddings(model='text-embedding-3-small')

    # store = Chroma.from_documents(documents=documents,embedding=embedding, persist_directory='./chromadb' ) 
    # store.persist() -> -> run this only the first time ; it will create files under the path mentioned in persist_directory


    load_to_disk = Chroma(persist_directory='./chromadb', embedding_function=embedding)

    

    query = 'what is TCP congestion control'

    ans = load_to_disk.similarity_search(query=query, k = 3)

    

    return ans, query



