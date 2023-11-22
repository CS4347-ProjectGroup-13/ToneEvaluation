import base64
from flask import Flask, render_template, request
from flask import request

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    print(request.files)
    audio = request.files['audio']
    print(audio)
    audio.save('audio.wav')
    # Process the audio file here
    # print(audio)
    return 'Audio file received'

if __name__ == '__main__':
    app.run(host = "localhost", port=12345, threaded=True)
