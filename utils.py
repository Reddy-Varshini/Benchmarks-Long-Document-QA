from langchain.text_splitter import RecursiveCharacterTextSplitter
import json 
from langchain.schema.document import Document
import os 
import torch
import math
import numpy as np


# # CHUNK SIZE PER MODEL
# Snowflake-xs --> 512
# Snowflake-large --> 512

# # Function to chunk the input document
def chunk_doc(inp, chunk_size = 2750):
    text = inp.replace("\\t"," ").replace("\t"," ").replace("\n\n"," ").replace("\t\t"," ").replace("\n"," ").replace("\\u2022","").replace("\u2022","").replace("\u2014","").replace("\\u2014","").replace("\\","")
    data = [Document(page_content=text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_size*0.2)
    all_chunks = text_splitter.split_documents(data)
    return all_chunks


# # Function to read documents
def get_doc(path):
    with open(path) as f:
        jd = json.load(f)
    return jd


# # Function to save embeddings
def save_embeddings(embeddings, folder_path, task_id):
    os.makedirs(folder_path+task_id, exist_ok=True)
    for i, doc in enumerate(embeddings):
        file_path = os.path.join(folder_path+task_id, f"doc_{i}.pt")
        torch.save(embeddings[i], file_path)


# # Function to load embeddings
def load_embeddings(folder_path):
    embeddings = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pt'):
            file_path = os.path.join(folder_path, file_name)
            embeddings.append(torch.load(file_path))
    return embeddings

