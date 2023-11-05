from dotenv import load_dotenv
import pandas as pd
import tiktoken
import openai
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from dotenv import load_dotenv


load_dotenv()  # take environment variables from .env.

 # move this to a secret manager after
secret_key = os.getenv('OPENAI_KEY')
organization = os.getenv('OPENAI_ORGANIZATION')


openai.api_key = secret_key
openai.Model.list()

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191


# load & inspect dataset
input_datapath = "data/output.csv"  # to save space, we provide a pre-filtered dataset
df = pd.read_csv(input_datapath, index_col=0)
df = df[["title",  "category", "description", "externalId"]]
df = df.dropna()

df["combined"] = (
    "Title: " + df.title.str.strip() + "; Category: " + df.category.str.strip() + "; Content: " + df.description.str.strip()
)

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

encoding = tiktoken.get_encoding(embedding_encoding)

df["embedding"] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
df.to_csv('data/embedding.csv', index=False)