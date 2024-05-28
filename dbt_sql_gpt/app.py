from flask import Flask, render_template
from flask_socketio import SocketIO, send
import os
from main_async import MyGPT
import asyncio
import json

loop = asyncio.get_event_loop()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)
chunks = []
chat = MyGPT(chunks=chunks)


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('message')
def handle_message(msg):
    async def llm_and_send():
        async for chunk in chat.run_llm_loop(chat.data_loader, msg):
            returned = chunk['messages'][0].content
            if returned == '':
                function_call = chunk['messages'][0].additional_kwargs['function_call']
                if function_call['name'] == 'run_query':
                    send(f"Executing Query {json.loads(function_call['arguments'])['query']}", broadcast=True)
            else:
                send(chunk['messages'][0].content, broadcast=True)

    loop.run_until_complete(llm_and_send())


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True, allow_unsafe_werkzeug=True)
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
