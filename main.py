from flask import Flask, request, render_template, jsonify
import os
import random
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

app = Flask(__name__)

# environment variables
_ = load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# user dataset
user_file_path = 'data/userdb.csv'
user_df = pd.read_csv(user_file_path)

# product dataset
product_file_path = 'data/productdb.csv'
product_df = pd.read_csv(product_file_path)

# Collaborative filtering
le = LabelEncoder()
le.fit(user_df['asin'])
user_df['asin_encoded'] = le.transform(user_df['asin'])

user_item_matrix = user_df.pivot_table(index='user_id', columns='asin_encoded', aggfunc='size', fill_value=0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_similar_users(user_id, num_users=5):
    """Get top N similar users to the given user_id."""
    if user_id not in user_similarity_df.index:
        raise ValueError(f"User {user_id} not found in the user similarity matrix.")
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:num_users+1]
    return similar_users

def recommend_products(user_id, num_recommendations=2):
    """Recommend products for a given user based on similar users' preferences."""
    similar_users = get_similar_users(user_id)
    similar_users_purchases = user_df[user_df['user_id'].isin(similar_users)]
    similar_users_purchases_grouped = similar_users_purchases.groupby('asin_encoded').size()
    user_purchases = set(user_df[user_df['user_id'] == user_id]['asin_encoded'])
    recommendations = similar_users_purchases_grouped[~similar_users_purchases_grouped.index.isin(user_purchases)]
    recommendations = recommendations.sort_values(ascending=False).head(num_recommendations).index
    recommended_asins = le.inverse_transform(recommendations)
    recommended_products = product_df[product_df['asin'].isin(recommended_asins)]
    return recommended_products[['title', 'url']]

# Random users for testing
random_users = ['user_80', 'user_81']
# Get titles of products purchased by random users for llm to use as context
user_titles = product_df[product_df['asin'].isin(user_df[user_df['user_id'].isin(random_users)]['asin'])]['title']

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                  model_kwargs={'device': 'cuda'}, encode_kwargs={'device': 'cuda'})

dataset_file_path = "data/productdb.csv"
loader = CSVLoader(file_path=dataset_file_path)
data = loader.load()

n_batch = 512
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model_path = "llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
n_gpu_layers = -1

template = f"""
[INST]
Context: {{context}}

You are a AI chatbot assistant for an e-commerce platform.
Your task is to recommend products to users based on their preferences using the context provided.
Always provide the url to the product and keep the answers as short as possible.
For example if the user asks: I want to buy a printer, what do you recommend?
You answer should be like the following and nothing more: I recommend Epson Expression Home XP-2105 Print/Scan/Copy Wi-Fi Printer, Black": https://www.amazon.co.uk/dp/B07VZJZX62 If you need further assistance, feel free to ask.

Try to use content-based filtering to recommend products based on the user's preferences from the previously purchased products which are as follows: {user_titles.to_list()}
Question: {{question}}
[/INST]
"""

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    temperature=0.0000000001,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=2000
)

vectordb = Chroma.from_documents(documents=data, embedding=embedding, persist_directory="./persist_directory")
retriever = vectordb.as_retriever()
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=1)

qa_prompt = PromptTemplate.from_template(template)
chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={'prompt': qa_prompt}
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load_model', methods=['POST'])
def load_model():
    global chain
    user_id = random.choice(random_users)
    num_recommendations = 2
    recommended_products = recommend_products(user_id, num_recommendations)

    recommendations = [
        {"title": row['title'], "url": row['url']}
        for index, row in recommended_products.iterrows()
    ]

    return jsonify({"status": "Model loaded", "recommendations": recommendations}), 200

@app.route('/chat', methods=['POST'])
def chat():
    if chain is None:
        return jsonify({"answer": "Model not loaded yet"}), 400

    question = request.form['question']
    result = chain({"question": question})

    answer = result['answer']

    return jsonify({"answer": answer}), 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
