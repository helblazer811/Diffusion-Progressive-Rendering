from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import emit
import time
from threading import Thread
import torch
from batched_lcm_pipeline import BatchedLCMPipeline
from batched_lcm_scheduler import BatchedLCMScheduler

app = Flask(__name__)

# Setup the neural network
model_id = "SimianLuo/LCM_Dreamshaper_v7"
pipe = BatchedLCMPipeline.from_pretrained(model_id, variant="fp16")
# pipe.load_lora_weights(lcm_lora_id)
pipe.scheduler = BatchedLCMScheduler.from_config(pipe.scheduler.config)
pipe.to(device="cuda", dtype=torch.float16)

# Make the callback
num_inference_steps = [1, 2, 3, 4]
# A generator function that yields results from a loop
def run_image_generator(prompt):
    yield from pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps, # This is the magic
        guidance_scale=1,
        # callback_on_step_end=callback,
    )
    # time.sleep(1)
    # yield f"Result {i}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def start_loop():
    prompt = request.form['prompt']
    # Start the loop in a separate thread to avoid blocking the server
    loop_thread = Thread(target=emit_loop_data, args=(prompt,))
    loop_thread.start()
    # return render_template('data.html')
    return render_template('index.html')

def emit_loop_data(prompt):
    for image in run_image_generator(prompt):
        print("Running generator")
        # Emit the data to connected clients using Socket.IO
        socketio.emit('image', {'image_data': image})
        time.sleep(1)

if __name__ == '__main__':
    from flask_socketio import SocketIO
    socketio = SocketIO(app)
    socketio.run(app, debug=True)