from loadDocs import load_docs
from dotenv import load_dotenv
# from langchain_community.embeddings import OpenAIEmbeddings
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

documents = load_docs()

# openaiKey = os.getenv("OPENAI-KEY")

# print(openaiKey)


embedding = OpenAIEmbeddings(model='text-embedding-3-small')

# store = Chroma.from_documents(documents=documents,embedding=embedding, persist_directory='./chromadb' )

# store.persist()

load_to_disk = Chroma(persist_directory='./chromadb', embedding_function=embedding)

# retriever = load_to_disk.as_retriever()

query = 'what is TCP congestion control'

ans = load_to_disk.similarity_search(query=query, k = 3)

for docs in ans:
    print(docs.page_content)

# print(ans)


