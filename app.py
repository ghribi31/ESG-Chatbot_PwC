from flask import Flask, request, render_template, jsonify
from ESGBOT.chatbot import ESGChatbot

app = Flask(__name__)

# Initialize the chatbot
model_path = r"D:\download\StagePwC\fine_tuned_model"
dataset_path = r"D:\download\StagePwC\ESGBOT\ESGBOT\Dataset\CompaniesDataESG.csv"
chatbot = ESGChatbot(model_path, dataset_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    req_data = request.get_json()
    query = req_data['query']
    chat_history = req_data.get('chat_history', [])
    answer = chatbot.query(query, chat_history)
    
    return jsonify({'answer': answer})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
