from openai import OpenAI
import numpy as np
import re 

path = "Long-Document_QA_Questions.json"

with open(path) as f:
    all_docs = json.load(f)


# # SELECT THE GPT MODEL TO RUN
client = OpenAI(api_key='')
model_checkpoint = "o1-preview" #"gpt-4-turbo" #"gpt-4o-2024-08-06"


def ask_gpt4(query, context):

    try:
    
        response = client.chat.completions.create(model=model_checkpoint,  
        messages=[
            {"role": "system", "content": "You are ChatGPT, a helpful assistant. You will answer queries with a single numerical value based on the provided context."},
            {"role": "user", "content": f"Context: {context}\nQuery: {query}\nAnswer with a single numerical value only."}
        ])

        answer = response.choices[0].message.content.strip()

        try:
            return answer
        except ValueError:
            return f"Error: Unable to parse numerical value from response: {answer}"

    except Exception as e:
        return f"Error: {str(e)}"

    
def ask_gpt_o(query, context):

    response = client.chat.completions.create(
        model=model_checkpoint,
        messages=[
            {
                "role": "user", 
                "content": f"Context: {trunc_context}\nQuestion: {query}\nAnswer as a single numerical value only."
            }
        ]
    )
    answer = response.choices[0].message.content.strip()

    try:
        return answer# if '.' in answer else int(answer)
    except ValueError:
        return f"Error: Unable to parse numerical value from response: {answer}"



task_id, model_response = [], []

for idx, doc in enumerate(all_docs):

    query = doc['question']
    context = doc['context']
    id = doc['id']

    if "o1" in model_checkpoint:
        response = ask_gpt_o(query, context)
    else:
        response = ask_gpt4(query, context)

    try:
        response = float(re.sub(r'[^0-9.-]', '', response))
    except:
        response = 0.0
    task_id.append(id)
    model_response.append(response)


# Create a DataFrame and save in required format
df = pd.DataFrame({'id': task_id, 'model_response': model_response})
df.to_csv(f'{model_checkpoint}-response.csv', index=False, header = False)
