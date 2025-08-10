#import goes there
from langchain_community import document_loaders
import os
from dotenv import load_dotenv
from langchain_text_splitters import  RecursiveCharacterTextSplitter

def load_docs():
#load the pdf not the txt
    loader = document_loaders.PyPDFLoader("paste the pdf file path") 

    #load the txt
    Text = loader.load()

    splitter_setting = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 50)

    splitted_text = splitter_setting.split_documents(Text)

    return splitted_text

