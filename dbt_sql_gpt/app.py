from flask import Flask, render_template
from flask_socketio import SocketIO, send
import os
from dbt_sql_gpt.base_serving import LLMFlavor
from dbt_sql_gpt.open_ai_serving import MyGPTOpenAI
from dbt_sql_gpt.ollama_serving import MyGPTOllama
import asyncio
import json
from flask import request

loop = asyncio.get_event_loop()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)
chats = {}


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def onConnect():
    currentSocketId = request.sid
    # chats[currentSocketId] = MyGPTOpenAI()
    chats[currentSocketId] = MyGPTOllama()

@socketio.on('disconnect')
def onDisconnect():
    currentSocketId = request.sid
    chats[currentSocketId] = None

@socketio.on('message')
def handle_message(msg):
    currentSocketId = request.sid
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
