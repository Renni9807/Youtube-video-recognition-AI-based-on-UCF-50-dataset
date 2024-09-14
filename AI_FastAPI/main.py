from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from tensorflow.keras.models import load_model
import cv2
from starlette.responses import StreamingResponse
from typing import List, Tuple
from pytube import YouTube
from collections import deque
import numpy as np
import shutil
import os
import zipfile
import shutil
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Form

app = FastAPI()

# initialize model & setting 
convlstm_model = load_model('model_code/convlstm_model.keras')
lrcn_model = load_model('model_code/LRCN_model.keras')
my_lrcn_model = load_model('model_code/my_LRCN_model.keras')
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace", "BaseballPitch", "Basketball", "BenchPress", "Biking", "Billiards", "Punch", "PushUps"]
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20

# create 'temp' folder if not exists
if not os.path.exists('temp'):
    os.makedirs('temp')
    
# This part is essential in order to prevent CORS error.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Requests from all domains are allowed. In actual service, it is recommended to configure settings to allow only specified domains.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all types of HTTP methods
    allow_headers=["*"],  # Allow all HTTP header
)

# YouTube video download function
def download_youtube_videos(youtube_url, output_path):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    video_title = stream.default_filename
    original_video_title = video_title
    video_number = 1
    while os.path.exists(os.path.join(output_path, video_title)):
        video_number += 1
        video_title = f"{original_video_title.split('.mp4')[0]}_{video_number}.mp4"
    stream.download(output_path=output_path, filename=video_title)
    return video_title

# extract video frame
def frames_extraction(video_path: str, sequence_length: int = SEQUENCE_LENGTH) -> List[np.ndarray]:
    """
    Extracts frames from a video file.
    
    Args:
    video_path (str): Path to the video file.
    sequence_length (int): The number of frames to extract for the sequence.

    Returns:
    List[np.ndarray]: A list of frames extracted from the video.
    """
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine the frame interval based on sequence length.
    if video_frames_count >= sequence_length:
        frame_interval = int(np.floor(video_frames_count / sequence_length))
    else:
        frame_interval = 1

    for frame_count in range(sequence_length):
        # Set the frame position.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
        success, frame = video_reader.read()
        if not success:
            break
        # Resize and normalize the frame.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)

    video_reader.release()
    return frames_list

def continuous_predict_and_overlay(input_video_path, model, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        sequence.append(normalized_frame)

        if len(sequence) == SEQUENCE_LENGTH:
            prediction_input = np.expand_dims(np.array(sequence), axis=0)
            predictions = model.predict(prediction_input)
            predicted_index = np.argmax(predictions, axis=1)[0]
            class_name = CLASSES_LIST[predicted_index]
            cv2.putText(frame, class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

@app.get("/")
def read_root():
    return {"Hello": "This is your FastAPI application"}
    

@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...)):
    input_video_path = "temp/input_video.mp4"
    output_video_path_conv = "temp/output_video_conv.mp4"
    output_video_path_lrcn = "temp/output_video_lrcn.mp4"
    output_video_path_my_lrcn = "temp/output_video_my_lrcn.mp4"

    with open(input_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    frames = frames_extraction(input_video_path)

    # analyze with selected model if model is chosen
    predictions_conv = convlstm_model.predict(np.expand_dims(frames, axis=0))
    predictions_lrcn = lrcn_model.predict(np.expand_dims(frames, axis=0))
    predictions_my_lrcn = my_lrcn_model.predict(np.expand_dims(frames, axis=0))

    predicted_class_names_conv = [CLASSES_LIST[i] for i in np.argmax(predictions_conv, axis=1)]
    predicted_class_names_lrcn = [CLASSES_LIST[i] for i in np.argmax(predictions_lrcn, axis=1)]
    predicted_class_names_my_lrcn = [CLASSES_LIST[i] for i in np.argmax(predictions_my_lrcn, axis=1)]

    # create zip file
    zip_path = "temp/analyzed_videos.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(output_video_path_conv, arcname=os.path.basename(output_video_path_conv))
        zipf.write(output_video_path_lrcn, arcname=os.path.basename(output_video_path_lrcn))
        zipf.write(output_video_path_my_lrcn, arcname=os.path.basename(output_video_path_my_lrcn))

    return FileResponse(zip_path, media_type="application/zip", filename="analyzed_videos.zip")


@app.post("/download-and-analyze-youtube/")
async def download_and_analyze_youtube(youtube_url: str = Form(...)):
    try:
        video_title = download_youtube_videos(youtube_url, 'temp')
        input_video_path = os.path.join('temp', video_title)
        output_video_path = f"temp/{video_title}_analyzed.mp4"
        
        # analyze video and over lay the prediction
        continuous_predict_and_overlay(input_video_path, my_lrcn_model, output_video_path)
        
        # convert analyzed video (not streaming but just file response) 
        return FileResponse(output_video_path, media_type="video/mp4")
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

@app.get("/stream-video/")
def stream_video():
    video_path = "path_to_your_video.mp4"  # video file root
    return StreamingResponse(open(video_path, "rb"), media_type="video/mp4")
@app.get("/stream-video/{video_name}")
async def stream_video(video_name: str):
    video_path = f"temp/{video_name}"  # video file root
    return StreamingResponse(open(video_path, "rb"), media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
