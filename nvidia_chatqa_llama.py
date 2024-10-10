from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import json
from utils import *
import time
import argparse
import os

# Load the retriever and context encoder models
retriever_tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder')
context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder')

# Function to process a single document using the RAG approach with batching
def process_document(doc, embedding_dir, chunk_size):

    chunked_doc = chunk_doc(doc['context'], chunk_size=chunk_size)
    documents = [chunks.page_content for chunks in chunked_doc]
    query = doc['question']

    messages = [
        {"role": "user", "content": query}
        ]

    formatted_query_for_retriever = '\n'.join([turn['role'] + ": " + turn['content'] for turn in messages]).strip()

    query_input = retriever_tokenizer(formatted_query_for_retriever, return_tensors='pt').to(device)
    query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]

    # Initialize list to store all top context and scores
    final_top_context = []
    final_top_scores = []

    batch_size = 400
    num_batches = (len(documents) + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch_docs = documents[i * batch_size:(i + 1) * batch_size]
        ctx_input = retriever_tokenizer(batch_docs, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]

        similarities = query_emb.matmul(ctx_emb.transpose(0, 1))  # (1, num_ctx)
        ranked_results = torch.argsort(similarities, dim=-1, descending=True)  # (1, num_ctx)

        # Get top context and scores for this batch
        batch_top_context = [batch_docs[idx] for idx in ranked_results.tolist()[0][:3]]

        try:
            batch_scores = similarities.squeeze().tolist()[:3]
            final_top_context.extend(batch_top_context)
            final_top_scores.extend(batch_scores)
        except:
            batch_scores = similarities.squeeze().tolist()
        # Extend the final context and scores list with the top from each batch
            final_top_context.append(batch_top_context)
            final_top_scores.append(batch_scores)

    # Sort the final top context based on the scores and keep the top results
    sorted_indices = torch.argsort(torch.tensor(final_top_scores), descending=True)
    top_context = [final_top_context[idx] for idx in sorted_indices[:1]]
    scores = [final_top_scores[idx] for idx in sorted_indices[:1]]

    return {
        "id": doc['id'], 
        "question": query, 
        "context": [(con, scores[idx]) for idx, con in enumerate(top_context)]
    }

# Main function to execute processing
def main():
    global model_checkpoint, tokenizer, model, embedding_dir, chunk_size, device

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process documents with Nvidia chatqa llama model.")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Model checkpoint to use.")
    parser.add_argument('--output_filename', type=str, required=True, help="File name to save the Nvidia chatqa llama response.")
    parser.add_argument('--chunk_size', type=int, required=True, help="Size of the document chunks.")


    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint
    output_filename = args.output_filename
    chunk_size = args.chunk_size

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float16, trust_remote_code=True)
    model.to(device)

    all_docs = get_doc("Long-Document_QA_Questions.json")
    
    embedding_dir = f"Embeddings/{model_checkpoint.replace('/', '_')}/"
    os.makedirs(embedding_dir, exist_ok=True)
    
    complete_run_start = time.time()

    # Process each document and write the result to the file
    for ii, doc in enumerate(all_docs):
        per_doc_start = time.time()
        result = process_document(doc, embedding_dir)

        # Append the result directly to the JSON file
        with open(output_filename, "a") as f:
            json.dump(result, f)
            f.write("\n")  # Write each result on a new line for easier reading

        print(f"{model_checkpoint}\t{ii+1}\t {round(time.time()-per_doc_start, 3)} seconds")


    del all_docs, model, tokenizer

    print(f"Total time {round(time.time()-complete_run_start, 3)} seconds")

if __name__ == "__main__":
    main()



# ** SAMPLE RUN CODE **
# CUDA_VISIBLE_DEVICES=0 python nvidia_chatqa_llama.py --model_checkpoint "nvidia/Llama3-ChatQA-1.5-8B" --output_filename "Data/nvidia-Llama3-ChatQA-1-5-8B-2048-tok.json" --chunk_size 8000
