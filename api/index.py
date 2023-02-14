import os
from flask import Flask, request, render_template

import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle

app = Flask(__name__)
os.environ.get("OPENAI_API_KEY")

# Load model
index = faiss.read_index("./data/docs.index")
with open("./data/faiss_store.pkl", "rb") as f:
    store = pickle.load(f)
store.index = index
chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)


@app.route("/")
def index():
    return "Hello!"


@app.route('/qa', methods=['GET', 'POST'])
def qa():
    if request.method == 'GET':
        return render_template('question.html')
    else:
        question = request.form['question'].strip()

        # Get the answer and sources
        result = chain({"question": question})
        return render_template('result.html', result=result)