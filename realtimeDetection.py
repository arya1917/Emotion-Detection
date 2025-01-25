import cv2
from keras.models import model_from_json
import numpy as np

json_file = open("EmotionDetection.json", "r")
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights("EmotionDetection.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image, dtype=np.float32)
    feature = feature.reshape(1, 48, 48, 1)  
    return feature / 255.0  

webcam = cv2.VideoCapture(0)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, frame = webcam.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))  
        face = extract_features(face)
        prediction = model.predict(face)
        label = np.argmax(prediction)  
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  
        cv2.putText(frame, labels[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2) 
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

webcam.release()
cv2.destroyAllWindows()
