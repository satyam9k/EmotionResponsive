# Emotion Recognition and Music Recommendation System

## Overview

The Emotion Recognition and Music Recommendation System uses facial and voice emotion recognition to recommend songs based on the detected emotions. It integrates various models for emotion detection from both video and audio sources, and plays a song that matches the detected emotion.

## Features

- Facial emotion recognition using a pre-trained model and OpenCV.
- Voice emotion recognition using Whisper and a pre-trained emotion prediction model.
- Music recommendation based on detected emotions.
- Real-time video and audio capture with emotion detection.
- Song playback with controls.

## Components

1. **Facial Emotion Recognition:**
   - Detects facial emotions using a pre-trained model.
   - Provides real-time feedback on detected emotions.

2. **Voice Emotion Recognition:**
   - Converts speech to text using a speech-to-text model.
   - Analyzes the text to detect the emotion in the voice.

3. **Music Recommendation:**
   - Recommends a song based on the detected emotion.
   - Plays the recommended song with user controls.

## Usage

1. **Load Models:**

    The models will be loaded automatically when the script runs.

2. **Run the Program:**

    Execute the script to start the application:

    ```bash
    python main.py
    ```

3. **Interact with the Program:**

    - **Facial Emotion Recognition:** Capture a 10-second video to detect facial emotions.
    - **Voice Emotion Recognition:** Record a 10-second audio clip to detect voice emotions.
    - **Complex Emotion Recognition:** Combine facial and voice emotion recognition for a more comprehensive result.
    - **Play Recommended Song:** Based on the detected emotion, a song will be recommended and played.

## Demo

You can view a demonstration of the Emotion Recognition and Music Recommendation System in action by watching the [demo video](https://github.com/satyam9k/EmotionResponsive/blob/main/demo.mp4).




## Acknowledgements

- Thanks to the developers of the pre-trained models used in this project.
- Special thanks to the contributors and maintainers of the libraries used.

