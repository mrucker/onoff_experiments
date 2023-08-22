from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

def embedder(items):
    if not isinstance(items,(str,list)):
        raise AssertionError()
    if isinstance(items,str):
        items = [items]
    elif isinstance(items,list):
        items = [ i['instruction'] for i in items]

    embeddings = model.encode(items,convert_to_tensor=True)
    normalized = torch.nn.functional.normalize(embeddings)

    return normalized