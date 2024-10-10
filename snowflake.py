import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
from utils import *
import json
import os
import time
from multiprocessing import set_start_method
import argparse

# Function to generate embeddings for a document
def generate_embedding(document, context_window=512):
    inputs = tokenizer(document, padding=True, truncation=True,
                       return_tensors='pt', max_length=context_window)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    embedding = model(**inputs)[0][:, 0]
    return embedding

# Function to find similar documents based on cosine similarity
def find_similar_documents(query_document, document_embeddings, documents, top_n=5):
    query_embedding = generate_embedding(query_document)
    similarity_scores = [cosine_similarity(query_embedding,
                                           doc_embedding).item() for doc_embedding in document_embeddings]
    sorted_indices = torch.argsort(torch.tensor(similarity_scores),
                                   descending=True)
    top_documents = [documents[idx] for idx in sorted_indices[:top_n]]
    top_scores = [similarity_scores[idx] for idx in sorted_indices[:top_n]]
    return top_documents, top_scores


# Function to process a single document and write the result to a JSON file
def process_and_save_document(doc, embedding_dir, output_filename, chunk_size):
    chunked_doc = chunk_doc(doc['context'], chunk_size=chunk_size)
    documents = [chunks.page_content for chunks in chunked_doc]
    del chunked_doc

    # Initialize list to store all top context and scores
    final_top_context = []
    final_top_scores = []
    
    batch_size = 400
    num_batches = (len(documents) + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch_docs = documents[i * batch_size:(i + 1) * batch_size]
        batch_embeddings = [generate_embedding(chunks, chunk_size) for chunks in batch_docs]

        batch_top_context, batch_scores = find_similar_documents(doc['question'], batch_embeddings, batch_docs, top_n=3)
        
        # Extend the final context and scores list with the top from each batch
        final_top_context.extend(batch_top_context)
        final_top_scores.extend(batch_scores)

    # Sort the final top context based on the scores and keep the top results
    sorted_indices = torch.argsort(torch.tensor(final_top_scores), descending=True)
    top_context = [final_top_context[idx] for idx in sorted_indices[:1]]
    scores = [final_top_scores[idx] for idx in sorted_indices[:1]]

    # Create the result dictionary
    result = {
        "id": doc['id'], 
        "question": doc['question'], 
        "context": [(con, scores[idx]) for idx, con in enumerate(top_context)]
    }
    
    # Write the result directly to a JSON file
    with open(output_filename, "a") as f:
        json.dump(result, f)
        f.write("\n") 

# Main function to execute multiprocessing
def main():
    global model_checkpoint, tokenizer, model, embedding_dir, chunk_size, device

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process documents with Snowflake model.")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Model checkpoint to use.")
    parser.add_argument('--output_filename', type=str, required=True, help="File name to save the Snowflake response.")
    parser.add_argument('--chunk_size', type=int, required=True, help="Size of the document chunks.")


    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint
    output_filename = args.output_filename
    chunk_size = args.chunk_size

    # Set the device
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModel.from_pretrained(model_checkpoint, add_pooling_layer=False, trust_remote_code=True)
    model.to(device)

    all_docs = get_doc("Data/long_document_w_answer.json")
    
    embedding_dir = f"Embeddings/{model_checkpoint.replace('/', '_')}/"
    os.makedirs(embedding_dir, exist_ok=True)
    
    complete_run_start = time.time()

    # Process each document individually and write the result to the file
    for ii, doc in enumerate(all_docs):
        per_doc_start = time.time()
        process_and_save_document(doc, embedding_dir, output_filename)
        print(f"{model_checkpoint}\t{ii}\t {round(time.time()-per_doc_start, 3)} seconds")

    torch.cuda.empty_cache()
    del all_docs, model, tokenizer

    print(f"Total time {round(time.time()-complete_run_start, 3)} seconds")

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    set_start_method('spawn')
    main()




# ** SAMPLE RUN CODE **
# CUDA_VISIBLE_DEVICES=0 python snowflake.py --model_checkpoint "Snowflake/snowflake-arctic-embed-xs" --output_filename "Data/snowflake-xs.json" --chunk_size 2750
