from flask import Flask, request
from search_embedding import get_results_for_query
app = Flask(__name__)

@app.route('/')
def get_query_search_results():
    search_string = request.args.get('search')
    print(search_string)
    results = get_results_for_query(search_string)
    return results

if __name__ == '__main__':
    app.run()