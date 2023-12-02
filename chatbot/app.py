import os

from flask import Flask, render_template, request, jsonify
import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair

vertexai.init(project="flaskserver-406902", location="us-central1")
chat_model = ChatModel.from_pretrained("chat-bison")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}
chat = chat_model.start_chat(
    context="""You are Cardiologist that  Evaluates low blood pressure issues concerning cardiovascular conditions and related concerns.""",
)
# pylint: disable=C0103
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    prompt = request.form["user_input"]
    response = chat.send_message(prompt, **parameters)
    print(f"Response from Model: {response.text}")
    return jsonify({"response": response.text})

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
