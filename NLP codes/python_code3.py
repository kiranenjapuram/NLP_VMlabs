import json
import torch
from transformers import BertTokenizer, BertModel
import numpy as np


# 3. Load the JSON file:
#    ```python
with open(r'C:\NLP codes\document.json', 'r') as file:
    json_data = json.load(file)


# 4. Load the BERT tokenizer and model:
#    ```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()


# 5. Create a function to generate embeddings for each JSON item:
#    ```python
def get_embeddings(data):
    embeddings = []
    for k,v in data.items():
        print(k,v)
        
        for p in v['pages']:
            page=p['content']
            inputs = tokenizer.encode_plus(page, add_special_tokens=True, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
                embedded_text = outputs.last_hidden_state.squeeze(0).mean(0).numpy()
                embeddings.append(embedded_text)

            
    return embeddings


# 6. Call the `get_embeddings` function with the loaded JSON data:
#    ```python
embeddings = get_embeddings(json_data)


# 7. Save the embeddings to a file:
#    ```python
np.savetxt('embeddings.txt', embeddings)