import io
import multiprocessing
import json

import numpy as np

from flask import Flask, render_template, request, send_file, send_from_directory
from PIL import Image
from shortest_path import find_shortest_path
from time import sleep
from threading import Thread

app = Flask(__name__)

raw_image = np.zeros([640, 480, 3])
dest_image = np.zeros([640, 480, 3])
work_done = False


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        global raw_image
        global dest_image
        file = request.files['file']
        img = Image.open(file.stream)
        raw_image = np.array(img)
        if raw_image.shape[2] > 3:
            raw_image = raw_image[:, :, :3]
        dest_image = raw_image
        return render_template('point_selector.html')
    return render_template('upload.html')

@app.route('/files/<filename>', methods=['GET'])
def raw_image(filename):
    global raw_image
    global dest_image
    imgs = {"raw_image.jpg": raw_image, "dest_image.jpg": dest_image}
    img = Image.fromarray(imgs.get(filename, np.zeros([640, 480, 3])))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    return send_file(img_buffer, mimetype='image/jpeg', as_attachment=True, download_name='image.jpg')

@app.route('/display.html', methods=['GET'])
def display():
    return render_template('display.html')

def worker(raw_image, x0, y0, x1, y1, conn):
    try:
        arr = find_shortest_path(raw_image, int(x0), int(y0), int(x1), int(y1))
        conn.send(tuple(map(lambda sr: tuple(map(tuple, sr)), arr)))
    except:
        conn.send((((0, 0, 0),),))

def start_work(x0, y0, x1, y1):
    global dest_image
    global work_done
    
    try:
        conn1, conn2 = multiprocessing.Pipe()
        process = multiprocessing.Process(target=worker, args=(raw_image, x0, y0, x1, y1, conn2))
        process.start()
        result = conn1.recv()
        dest_image = np.array(result).astype(np.uint8)
        while process.is_alive():
            sleep(0.1)
        process.join()
        process.exitcode
    finally:
        work_done = True

@app.route('/calculate_path', methods=['GET'])
def calculate_path():
        
    x0 = request.args.get('x0') 
    y0 = request.args.get('y0') 
    x1 = request.args.get('x1') 
    y1 = request.args.get('y1') 
    
    global work_done
    work_done = False
    Thread(target=lambda: start_work(x0, y0, x1, y1)).start()
    return "ok"

@app.route('/is_work_done', methods=['GET'])
def is_work_done():
    global work_done
    return json.dumps({"work_done": work_done})
    
@app.route('/await.html', methods=['GET'])
def await_html():
    global work_done
    return render_template('await.html')
    
@app.route('/waiting-icon.gif')
def download_waiting_icon():
    return send_from_directory('static', 'waiting-icon.gif', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
