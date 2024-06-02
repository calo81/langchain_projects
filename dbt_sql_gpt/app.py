import asyncio
import json
import os
import uuid

from flask import Flask, render_template, request, session
from flask_socketio import SocketIO, send

from dbt_sql_gpt.base_serving import LLMFlavor
from dbt_sql_gpt.ollama_serving import MyGPTOllama, OllamaFlavor
from dbt_sql_gpt.open_ai_serving import MyGPTOpenAI
from flask_session import Session

loop = asyncio.get_event_loop()

app = Flask(__name__)

app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
socketio = SocketIO(app)
chats = {}
model_in_use = {}


@app.route('/')
def index():
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload_files', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return 'No files part', 400
    files = request.files.getlist('files[]')
    for file in files:
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return 'Files successfully uploaded', 200

@app.route('/set_openai', methods=['GET'])
def set_openai():
    model_in_use[session['session_id']] = LLMFlavor.OpenAI
    chats[session['session_id']] = MyGPTOpenAI()
    return "Using OpenAI", 200

@app.route('/set_ollama', methods=['GET'])
def set_ollama():
    model_in_use[session['session_id']] = LLMFlavor.Ollama
    chats[session['session_id']] = MyGPTOllama(OllamaFlavor.codestral)
    return "Using Ollama Codestral", 200

@socketio.on('connect')
def onConnect():
    if session.get('session_id') is None:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    currentSocketId = session['session_id']
    chats[currentSocketId] = MyGPTOpenAI()
    # chats[currentSocketId] = MyGPTOllama(OllamaFlavor.codestral)

@socketio.on('disconnect')
def onDisconnect():
    currentSocketId = session['session_id']
    chats[currentSocketId] = None

@socketio.on('message')
def handle_message(msg):
    currentSocketId = session['session_id']
    chat = chats[currentSocketId]
    async def llm_and_send():
        async for chunk in chat.run_llm_loop(chat.data_loader, msg):
            if hasattr(chunk, 'content'):
                send(chunk.content)
            else:
                returned = chunk['messages'][0].content
                if returned == '':
                    function_call = chunk['messages'][0].additional_kwargs['tool_calls'][0]['function']
                    if function_call['name'] == 'run_query':
                        send(f"Executing Query {json.loads(function_call['arguments'])['query']}")
                else:
                    send(chunk['messages'][0].content)

    loop.run_until_complete(llm_and_send())


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True, allow_unsafe_werkzeug=True)
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
