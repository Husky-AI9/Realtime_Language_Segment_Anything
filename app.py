import os
import warnings
import gradio as gr
import numpy as np
from PIL import Image
from llm import efficientViT_SAM  
from webcam import *
from ultralytics import YOLO
import time

warnings.filterwarnings("ignore")
sam_model = efficientViT_SAM()  # Example model; adjust as needed
realtime_model = Realtime_Lang_Sam(sam_model)
yolo_world_m = YOLO('yolov8x-world.pt')

def predict(conf_threshold, iou_threshold, input, text_prompt):
    text_prompt = text_prompt.split(',')
    realtime_model.init_model_medium(prompt=text_prompt,model=yolo_world_m)
    output = realtime_model.predict_frame(input,conf_threshold,iou_threshold)
    return output 

def video_predict(conf_threshold, iou_threshold, input, text_prompt):
    text_prompt = text_prompt.split(',')
    realtime_model.init_model_medium(prompt=text_prompt,model=yolo_world_m)
    return realtime_model.predict_video(input, conf_threshold, iou_threshold)

def realtime_predict(prompt):
    prompt = prompt.split(',')
    realtime_model.init_model(prompt)
    realtime_model.predict_realtime()


# Define the input components directly from the 'gr' namespace
inputs = [
    gr.Slider(minimum=0, maximum=1, value=0.3, label="CONF threshold"),
    gr.Slider(minimum=0, maximum=1, value=0.25, label="IOU threshold"),
    gr.Image(label='Image', type='pil'),  # Now expecting a PIL Image directly
    gr.Textbox(label="Text Prompt", lines=2, placeholder="Enter text here..."),
]

inputs2 = [
    gr.Slider(minimum=0, maximum=1, value=0.3, label="CONF threshold"),
    gr.Slider(minimum=0, maximum=1, value=0.25, label="IOU threshold"),
    gr.Video(),
    gr.Textbox(label="Text Prompt", lines=2, placeholder="Enter text here..."),
]


# Define the output component directly from the 'gr' namespace
output = gr.Image(label="Output Image")
output2 = gr.Video(label="Output Video")
# Example data
examples = [
    [0.36, 0.25, "assets/fig/cat.jpg", "cat"],
    [0.36, 0.25, "assets/demo/fruits.jpg", "bowl"],
]


with gr.Blocks() as app3:
    gr.Markdown(
    """
    # Language-Segment-Anything Realtime!
    """)
    with gr.Row():
        prompt_input = gr.Textbox(label="Enter Prompt", placeholder="Type your prompt here...")
        start_button = gr.Button("Start Processing")



    # Define the action to take when the start button is clicked
    start_button.click(fn=realtime_predict, inputs=prompt_input, concurrency_id="fn")

# Create the interface
app1 = gr.Interface(fn=predict, inputs=inputs, outputs=output, examples=examples, title="Language-Segment-Anything Photo Prediction", description="Generates predictions using the LangSAM model.")

app2 = gr.Interface(fn=video_predict,inputs=inputs2,outputs=output2,title="Language-Segment-Anything Video Prediction")
# Launch the interface
demo = gr.TabbedInterface([app1, app2, app3], ["Image", "Video", "Webcam"])
demo.queue(default_concurrency_limit=1)

demo.launch()