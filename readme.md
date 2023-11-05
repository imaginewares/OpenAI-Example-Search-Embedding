# Example Application to create AI powered search for your website using OpenAI ChatGPT embeddings

## Introduction

This application demonstrates how you can easily implement OpenAI powered search on your database quickly with little setup.
It uses text-embedding-ada-002 model and cosine similarity distance function. More info and alternatives can be found (https://platform.openai.com/docs/guides/embeddings)[here].

## Steps to install and run

- This sample uses Python version Python3.10.9 and python3.10-venv 
- Create a new virtual environment
```
python3.10 -m venv venv
source ./venv/bin/activate
```

- Install all items in requirements.txt

`pip install -r requirements.txt``

- Put your datasource csv as data/input.csv

- Create an env file as in .env.example. You will need to obtain OpenAI API key from platform.openai.com

- Replace columns in create_embedding.py with column names in your input.csv dataset that you want to search against.

- Create the embedding python create_embedding.py. Wait for the process to finish. You should now see data/embeddings.csv
For large file size embeddings use external data-store for faster indexing.

-  Run the Flask server python main.py
Query the embedding http://127.0.0.1:5000/?search={searchString}

## That's it you now have an AI powered search against your production database for your website.

