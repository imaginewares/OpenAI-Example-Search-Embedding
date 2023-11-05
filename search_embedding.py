from dotenv import load_dotenv



import pandas as pd
import tiktoken
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from ast import literal_eval

from openai.embeddings_utils import get_embedding
import json


load_dotenv()  # take environment variables from .env.

secret_key = os.getenv('OPENAI_KEY')
organization = os.getenv('OPENAI_ORGANIZATION')

openai.organization = organization
openai.api_key = secret_key

embeddingFilePath = "data/embedding.csv"
embedding_model = "text-embedding-ada-002"

df = pd.read_csv('data/embedding.csv')
df['embedding'] = df.embedding.apply(literal_eval).apply(np.array)

def get_embedding(text, model=embedding_model):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


def search_products(df, search_string, n=3):
    search_embedding = get_embedding(search_string, model=embedding_model)
    df['similarities'] = df.embedding.apply(lambda x: cosine_similarity(x.reshape(1, -1), np.array(search_embedding).reshape(1, -1))[0][0])
    res = df.sort_values('similarities', ascending=False).head(n)
    return res


def get_results_for_query(query, n=3):
    res = search_products(df, query, n=n)
    json_obj = res.to_json(orient='records')
    return json_obj
