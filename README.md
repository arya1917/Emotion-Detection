---

# Real-Time Face Emotion Detection System 🎭  

This project is a **real-time face emotion detection system** built using **Python**, **OpenCV**, and **Keras**. It utilizes a pre-trained deep learning model to classify facial emotions from real-time webcam input, enabling applications in various fields such as mental health monitoring and user experience enhancement.  

---

## Features ✨  
- **Real-Time Detection**: Detects emotions from live webcam video feed.  
- **Deep Learning Model**: Leverages a pre-trained model (`EmotionDetection.h5`) for accurate emotion classification.  
- **Emotion Categories**: Identifies emotions such as angry, happy, sad, neutral, surprise, and more.  

---

## Technologies Used 🛠️  
- **Python**: Core programming language for development.  
- **OpenCV**: For face detection and real-time video processing.  
- **Keras** with TensorFlow backend: To load and run the emotion detection model.  
- **Matplotlib**: For visualizing training data (if needed).  

---

## File Structure 📁  
- **`EmotionDetection.h5`**: Pre-trained weights for the deep learning model.  
- **`EmotionDetection.json`**: Model architecture in JSON format.  
- **`realtimeDetection.py`**: Main script for real-time emotion detection.  
- **`main.ipynb`**: Jupyter Notebook for additional testing and experiments.  
- **`requirements.txt`**: List of Python dependencies.  

---

## How to Run the Project 🚀  

1. **Clone the Repository**:  
   ```bash  
   git clone https://github.com/arya1917/Emotion-Detection.git  
   cd Emotion-Detection  
   ```  

2. **Install Dependencies**:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Run Real-Time Detection**:  
   ```bash  
   python realtimeDetection.py  
   ```  
   - This will activate your webcam, detect faces in the video, and display the identified emotions on the screen.  

---

## Emotion Categories Recognized 🧠  
- **Angry**  
- **Disgust**  
- **Fear**  
- **Happy**  
- **Neutral**  
- **Sad**  
- **Surprise**  

---

## Code Highlights ⚙️  

1. **Model Loading**:  
   The model is loaded from JSON and pre-trained weights:  
   ```python  
   json_file = open("EmotionDetection.json", "r")  
   model_json = json_file.read()  
   json_file.close()  
   model = model_from_json(model_json)  
   model.load_weights("EmotionDetection.h5")  
   ```  

2. **Real-Time Detection**:  
   Detects faces in a webcam feed, extracts features, and predicts emotions:  
   ```python  
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
   faces = face_cascade.detectMultiScale(gray, 1.3, 5)  
   for (x, y, w, h) in faces:  
       face = gray[y:y + h, x:x + w]  
       face = cv2.resize(face, (48, 48))  
       face = extract_features(face)  
       prediction = model.predict(face)  
       label = np.argmax(prediction)  
   ```  

---

## Future Enhancements 🛠️  
- Train the model on a larger dataset for improved accuracy.  
- Add more nuanced emotion categories for better classification.  
- Develop a GUI for easier interaction.  
- Integrate with external APIs for advanced applications (e.g., mood-based music).  

---

## Credits 🙌  
Developed by **Arya Subhash Jadhav**.  
Feel free to fork, improve, and contribute to this project!  

---

## License 📄  
This project is open-source and available under the MIT License.  

