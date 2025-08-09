#import goes there
from langchain_community import document_loaders
import os
from dotenv import load_dotenv
# from langchain.chat_models import ollama
from langchain_text_splitters import  RecursiveCharacterTextSplitter

def load_docs():
#load the pdf not the txt
    loader = document_loaders.PyPDFLoader("./data/TCPalgo.pdf") 

    #load the txt
    Text = loader.load()


    # text = "Hi my name is ghost, I write programs or atlest try to write"

    # docs = loader.load_and_split()

    splitter_setting = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 50)

    splitted_text = splitter_setting.split_documents(Text)

    return splitted_text
# print(len(splitted_text), type(splitted_text))

chucks = load_docs()
print(len(chucks))

