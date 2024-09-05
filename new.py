import os
import cv2
import numpy as np
import pandas as pd
import pygame
import threading
import torch
import soundfile as sf
import sounddevice as sd
import time
import keyboard
from tensorflow import keras
from keras.models import model_from_json, Sequential
from keras.utils import img_to_array
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification

# Global variables
classifier = None
face_cascade = None
speech_to_text_model = None
speech_to_text_processor = None
emotion_model = None
emotion_tokenizer = None
bert_model = None
bert_tokenizer = None

emotion_name = {0: 'happy', 1: 'neutral', 2: 'sad', 3: 'surprise'}

def load_all_models():
    global classifier, face_cascade, speech_to_text_model, speech_to_text_processor, emotion_model, emotion_tokenizer, bert_model, bert_tokenizer

    # Load facial emotion recognition model
    with open('D:/Projects/Specialization_Project/Emotion-Recognition--main/models/emotion_model1.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    classifier = model_from_json(loaded_model_json, custom_objects={'Sequential': Sequential})
    classifier.load_weights("D:/Projects/Specialization_Project/Emotion-Recognition--main/models/emotion_model1.h5")

    # Load face cascade
    face_cascade = cv2.CascadeClassifier('D:/Projects/Specialization_Project/Emotion-Recognition--main/models/haarcascade_frontalface_default.xml')

    # Load Speech-to-Text Model
    speech_to_text_model = WhisperForConditionalGeneration.from_pretrained("D:/Projects/Specialization_Project/SER/models/speech-to-text-model")
    speech_to_text_processor = WhisperProcessor.from_pretrained("D:/Projects/Specialization_Project/SER/models/speech-to-text-processor")

    # Load Emotion Prediction Model for voice
    emotion_model = AutoModelForSequenceClassification.from_pretrained("D:/Projects/Specialization_Project/SER/models/emotion-prediction-model")
    emotion_tokenizer = AutoTokenizer.from_pretrained("D:/Projects/Specialization_Project/SER/models/emotion-prediction-tokenizer")

    # Load BERT model for song recommendation
    bert_model = BertForSequenceClassification.from_pretrained("D:/Projects/Specialization_Project/Song_DB/saved_modelv2")
    bert_tokenizer = BertTokenizer.from_pretrained("D:/Projects/Specialization_Project/Song_DB/saved_modelv2")

    print("All models loaded successfully.")

def detect_emotion(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
    emotion_output = "No face detected"

    for (x, y, w, h) in faces:
        roi_gray = img_gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = classifier.predict(roi)[0]
            
            prediction_adjusted = np.delete(prediction, 0)
            maxindex = int(np.argmax(prediction_adjusted))
            emotion_output = emotion_name[maxindex]

            # Draw rectangle around face and emotion text
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, emotion_output, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return emotion_output, image

def record_audio(output_file_path, duration=5, samplerate=16000):
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    sf.write(output_file_path, recording, samplerate, format='FLAC')

def predict_speech_to_text(audio_file_path, output_text_path):
    try:
        audio_input, samplerate = sf.read(audio_file_path)
        input_features = speech_to_text_processor(audio_input, return_tensors="pt", sampling_rate=samplerate).input_features
        
        predicted_ids = speech_to_text_model.generate(input_features, forced_decoder_ids=speech_to_text_processor.get_decoder_prompt_ids(language="en"))
        transcription = speech_to_text_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        with open(output_text_path, "w") as file:
            file.write(transcription)
        return transcription
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

def predict_emotion(text):
    emotion_mapping = {
        "anger": "sad", "neutral": "neutral", "sadness": "sad", "joy": "happy",
        "fear": "sad", "disgust": "sad", "surprise": "surprise"
    }
    try:
        inputs = emotion_tokenizer(text, return_tensors="pt")
        predictions = emotion_model(**inputs).logits
        predicted_class_id = torch.argmax(predictions, axis=-1).item()
        emotion = emotion_model.config.id2label[predicted_class_id]

        return emotion_mapping.get(emotion, "neutral")
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return None

def play_song(song_path, control):
    pygame.mixer.init()
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()
    
    while not control['stop']:
        command = input("Enter 's' to stop: ")
        if command.lower() == 's':
            pygame.mixer.music.stop()
            control['stop'] = True
            print("Playback stopped.")
        else:
            print("Invalid input. Please enter 's' to stop.")


def capture_video_and_audio(duration=10):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # Audio setup
    audio_duration = duration  # in seconds
    sample_rate = 16000  # standard sample rate
    audio_recording = np.empty((0, 1), dtype=np.float32)

    start_time = time.time()
    emotions = []
    recording = False

    def audio_callback(indata, frames, time, status):
        nonlocal audio_recording
        if status:
            print(status)
        audio_recording = np.append(audio_recording, indata, axis=0)

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate):
        while True:
            ret, frame = cap.read()
            if ret:
                emotion, annotated_frame = detect_emotion(frame)
                
                if recording:
                    out.write(frame)
                    if emotion != "No face detected":
                        emotions.append(emotion)
                    
                    elapsed_time = int(time.time() - start_time)
                    cv2.putText(annotated_frame, f"Recording: {elapsed_time}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if elapsed_time >= duration:
                        recording = False
                        break

                cv2.imshow('Facial Emotion Recognition', annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                elif keyboard.is_pressed('c') and not recording:
                    recording = True
                    start_time = time.time()
                    print("Starting 10-second video and audio capture...")
                    audio_recording = np.empty((0, 1), dtype=np.float32)  # Reset audio recording

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save the audio recording
    sf.write('recorded_audio.flac', audio_recording, sample_rate)

    # Determine the most frequent emotion during the video
    if emotions:
        most_frequent_emotion = max(set(emotions), key=emotions.count)
    else:
        most_frequent_emotion = None

    return most_frequent_emotion

def video_emotion_recognition():
    print("Starting 10-second video capture for emotion recognition...")
    most_frequent_facial_emotion = capture_video_and_audio(duration=10)

    try:
        transcription_text = predict_speech_to_text("recorded_audio.flac", "transcription.txt")
        if transcription_text:
            print(f"Transcription: {transcription_text}")
            voice_emotion = predict_emotion(transcription_text)
            print(f"Voice Emotion: {voice_emotion}")
        else:
            voice_emotion = None
    except Exception as e:
        print(f"Error processing audio file: {e}")
        voice_emotion = None

    return most_frequent_facial_emotion, voice_emotion

def complex_emotion_recognition_from_video():
    facial_emotion, voice_emotion = video_emotion_recognition()
    
    print(f"Detected Facial Emotion: {facial_emotion}")
    print(f"Detected Voice Emotion: {voice_emotion}")
    
    if facial_emotion == 'happy':
        if voice_emotion == 'happy':
            return 'happy'
        elif voice_emotion == 'sad':
            return 'sad'
        elif voice_emotion == 'neutral':
            return 'happy'
        elif voice_emotion == 'surprise':
            return 'surprise'
    elif facial_emotion == 'sad':
        if voice_emotion == 'happy':
            return 'happy'
        elif voice_emotion == 'sad':
            return 'sad'
        elif voice_emotion == 'neutral':
            return 'neutral'
        elif voice_emotion == 'surprise':
            return 'neutral'
    elif facial_emotion == 'neutral':
        if voice_emotion == 'happy':
            return 'happy'
        elif voice_emotion == 'sad':
            return 'sad'
        elif voice_emotion == 'neutral':
            return 'neutral'
        elif voice_emotion == 'surprise':
            return 'surprise'
    elif facial_emotion == 'surprise':
        if voice_emotion == 'happy':
            return 'surprise'
        elif voice_emotion == 'sad':
            return 'neutral'
        elif voice_emotion == 'neutral':
            return 'neutral'
        elif voice_emotion == 'surprise':
            return 'surprise'

    # Default behavior if one or both emotions are not detected
    if facial_emotion:
        return facial_emotion
    elif voice_emotion:
        return voice_emotion
    else:
        return 'neutral'  # Default to neutral if no emotion is detected

def play_recommended_song(emotion):
    download_dir = "D:/Projects/Specialization_Project/Song_DB/downloaded_songsv2"
    csv_file_path = 'D:/Projects/Specialization_Project/Song_DB/selected_tracks_info.csv'
    df = pd.read_csv(csv_file_path)

    emotion_map = {'sad': 0, 'happy': 1, 'neutral': 3, 'surprise': 2}
    if emotion in emotion_map:
        emotion_label = emotion_map[emotion]
        df_emotion = df[df['label'] == emotion_label]
        if df_emotion.empty:
            print(f"No songs found for the emotion: {emotion}")
            return

        random_song = df_emotion.sample(1).iloc[0]
        track_name = random_song['track_name']
        song_path = os.path.join(download_dir, f"{track_name}.mp3")
        
        if os.path.exists(song_path):
            print(f"Emotion detected is {emotion}, playing '{track_name}' by {random_song['artists']}")
            control = {'stop': False}
            play_thread = threading.Thread(target=play_song, args=(song_path, control))
            play_thread.start()
            play_thread.join()
        else:
            print(f"The song '{track_name}' for emotion '{emotion}' is not available in the directory.")
    else:
        print(f"Emotion '{emotion}' not recognized for song recommendation.")

def main_menu():
    while True:
        print("\nEmotion Recognition and Music Recommendation System")
        print("1. Facial Emotion Recognition")
        print("2. Voice Emotion Recognition")
        print("3. Complex Emotion Recognition")
        print("4. Quit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            facial_emotion = capture_video_and_audio(duration=10)
            print(f"Detected Facial Emotion: {facial_emotion}")
            if facial_emotion:
                play_recommended_song(facial_emotion)
        elif choice == '2':
            print("Recording 10-second audio...")
            record_audio("recorded_audio.flac", duration=10)
            transcription_text = predict_speech_to_text("recorded_audio.flac", "transcription.txt")
            if transcription_text:
                voice_emotion = predict_emotion(transcription_text)
                print(f"Detected Voice Emotion: {voice_emotion}")
                if voice_emotion:
                    play_recommended_song(voice_emotion)
        elif choice == '3':
            complex_emotion = complex_emotion_recognition_from_video()
            print(f"Detected Complex Emotion: {complex_emotion}")
            if complex_emotion:
                play_recommended_song(complex_emotion)
        elif choice == '4':
            print("Exiting the program...")
            break
        else:
            print("Invalid choice. Please try again.")
            continue
        
if __name__ == "__main__":
    load_all_models()
    main_menu()
