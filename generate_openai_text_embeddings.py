import openai
import torch
from transformers import AutoTokenizer, AutoModel
from utils import *
import json
import os
import time
from multiprocessing import set_start_method
import argparse
from openai import OpenAI
client = OpenAI()


def generate_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    embed = client.embeddings.create(input = [text], model=model).data[0].embedding
    return embed


def save_embeddings(embedding, path):
    with open(path, "w") as f:
        json.dump(embedding, f)



# Function to process a single document and write the result to a JSON file
def process_document(doc, embedding_dir, chunk_size):

    chunked_doc = chunk_doc(doc['context'], chunk_size=chunk_size)
    documents = [chunks.page_content for chunks in chunked_doc]
    del chunked_doc

    base_path = f"{embedding_dir}/{chunk_size}/{doc['id']}/"
    os.makedirs(base_path, exist_ok=True)

    for idx, text in enumerate(documents):
        embedding = generate_embedding(text, model_checkpoint)
        save_embeddings(embedding,  f"{base_path}{idx}.json" )

    embedding = generate_embedding(doc['question'], model_checkpoint)
    save_embeddings(embedding,  f"{base_path}query_embedding.json" )


# Main function to execute multiprocessing
def main():
    global model_checkpoint, embedding_dir, chunk_size, device

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process documents with OpenAI text embedding model.")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Model checkpoint to use.")
    parser.add_argument('--chunk_size', type=int, required=True, help="Size of the document chunks.")

    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint
    chunk_size = args.chunk_size


    # Set the device
    device = torch.device("cuda")

    all_docs = get_doc("Long-Document_QA_Questions.json")


    complete_run_start = time.time()

    embedding_dir = f"Embeddings/{model_checkpoint.replace('/', '_')}/"
    os.makedirs(embedding_dir, exist_ok=True)

    for ii, doc in enumerate(all_docs):
        per_doc_start = time.time()
        process_document(doc, embedding_dir)
        print(f"{model_checkpoint}\t{ii+1}\t {round(time.time()-per_doc_start,3)}")

    print(f"Total time {round(time.time()-complete_run_start, 3)} seconds")

if __name__ == "__main__":

    main()
