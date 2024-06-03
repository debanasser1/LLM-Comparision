import requests
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import openai
import pinecone
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from concurrent.futures import ThreadPoolExecutor
import logging
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# logging setup
logging.basicConfig(level=logging.INFO)

# pinecone setup
api_key = ‘xxxxxxxxxxxxxxxxxx’
pinecone.init(api_key=api_key, environment='us-west1-gcp')
index_name = 'web-scraped-data'

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean'
    )
index = pinecone.Index(index_name)

# OpenAI and Replicate API keys
openai.api_key = 'xxxxxxxxxxxxxx'
replicate_api_key = 'xxxxxxxxxxxxx'

#generate vector from text
def generate_vector(text):
    return [float(ord(c)) for c in text[:1536]]

# web scraping
def scrape_website(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text()
        return text_content
    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping {url}: {e}")
        return None

# indexing of scraped data
def index_data(urls):
    vectors = []
    for url in urls:
        content = scrape_website(url)
        if content:
            vector = generate_vector(content)
            vectors.append({"id": url, "values": vector})
    index.upsert(vectors)

# querying of Pinecone
def query_pinecone(query):
    query_vector = generate_vector(query)
    response = index.query(query_vectors=[query_vector], top_k=5)
    return [match['id'] for match in response['matches']]

# querying of GPT-3.5-turbo
def query_gpt_3_5_turbo(prompt):
    response = openai.Completion.create(
        model='gpt-3.5-turbo',
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# querying of GPT-4
def query_gpt_4(prompt):
    response = openai.Completion.create(
        model='gpt-4',
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# querying of Llama-2-70b-chat
def query_llama_2_70b_chat(prompt):
    response = requests.post(
        'https://api.replicate.com/v1/predictions',
        json={'version': 'llama-2-70b-chat', 'input': {'prompt': prompt}},
        headers={'Authorization': f'Token {replicate_api_key}'}
    )
    return response.json()['output']

# querying of Falcon-40b-instruct
def query_falcon_40b_instruct(prompt):
    response = requests.post(
        'https://api.replicate.com/v1/predictions',
        json={'version': 'falcon-40b-instruct', 'input': {'prompt': prompt}},
        headers={'Authorization': f'Token {replicate_api_key}'}
    )
    return response.json()['output']

# Function to measure answer relevancy (evaluate the LLM)
def measure_relevancy(input_prompt, response, context):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(input=input_prompt, actual_output=response, retrieval_context=context)
    answer_relevancy_metric.measure(test_case)
    return {
        'score': answer_relevancy_metric.score,
        'reason': answer_relevancy_metric.reason,
        'is_successful': answer_relevancy_metric.is_successful()
    }

#  best model based on relevancy scores
def print_best_model(scores):
    best_model = max(scores, key=lambda x: x['relevancy_score']['score'])
    logging.info(f"Best Model: {best_model['model_name']} with score {best_model['relevancy_score']['score']}")
    return best_model

# Flask endpoint to handle queries and determine the best model
@app.route('/query', methods=['POST'])
def handle_query():
    user_prompt = request.json['prompt']
    search_results = query_pinecone(user_prompt)
    combined_prompt = f"{user_prompt}\n{search_results}"

    with ThreadPoolExecutor() as executor:
        futures = {
            'gpt-3.5-turbo': executor.submit(query_gpt_3_5_turbo, combined_prompt),
            'gpt-4': executor.submit(query_gpt_4, combined_prompt),
            'llama-2-70b-chat': executor.submit(query_llama_2_70b_chat, combined_prompt),
            'falcon-40b-instruct': executor.submit(query_falcon_40b_instruct, combined_prompt),
        }

        responses = {model: future.result() for model, future in futures.items()}

    scores = [
        {
            'model_name': model,
            'response': response,
            'relevancy_score': measure_relevancy(user_prompt, response, search_results)
        }
        for model, response in responses.items()
    ]

    best_model = print_best_model(scores)
    emit('best_model', {'model_name': best_model['model_name'], 'score': best_model['relevancy_score']['score']})
    return jsonify(scores=scores)

if __name__ == '__main__':
    urls = [
        'https://u.ae/en/information-and-services',
        'https://u.ae/en/information-and-services/visa-and-emirates-id',
        'https://u.ae/en/information-and-services/visa-and-emirates-id/residence-visas',
        'https://u.ae/en/information-and-services/visa-and-emirates-id/residence-visas/golden-visa'
    ]
    index_data(urls)
    logging.info("Indexing completed. Starting Flask app.")
    socketio.run(app, debug=True)
