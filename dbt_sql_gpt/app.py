from flask import Flask, render_template
from flask_socketio import SocketIO, send
import os
from main import MyGPT

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)
chat = MyGPT(static_context="sql schema: cosas")
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(msg):
    print(f"Message: {msg}")
    result = chat.run_llm_loop(chat.data_loader, msg)
    send(result['output'], broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)