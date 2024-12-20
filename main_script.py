import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import Normalizer
import joblib


# Paths
FACES_DIR = 'faces/'
ENCODINGS_FILE = 'face_data.pkl'

# Ensure the faces directory exists
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

# Load saved face encodings
if os.path.exists(ENCODINGS_FILE):
    data = joblib.load(ENCODINGS_FILE)
    known_encodings = data['encodings']
    known_names = data['names']
else:
    known_encodings = []
    known_names = []

# Function to capture and save face images
def capture_and_save_face(folder_name):
    folder_path = os.path.join(FACES_DIR, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cap = cv2.VideoCapture(0)
    print(f"Capturing images for {folder_name}... Press 'c' to capture an image and 'q' to quit.")
    image_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow(f"Capturing for {folder_name}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(faces) > 0:
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                filename = os.path.join(folder_path, f"{folder_name}_{image_count + 1}.jpg")
                cv2.imwrite(filename, face)
                print(f"Image {image_count + 1} saved for {folder_name}")
                image_count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to encode faces in a folder
def encode_faces():
    global known_encodings, known_names
    for folder in os.listdir(FACES_DIR):
        folder_path = os.path.join(FACES_DIR, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.jpg') or file.endswith('.png'):
                    file_path = os.path.join(folder_path, file)
                    encoding = DeepFace.represent(file_path, model_name='Facenet', enforce_detection=False)[0]['embedding']
                    known_encodings.append(Normalizer().fit_transform([encoding])[0])
                    known_names.append(folder)

    # Save the updated encodings
    joblib.dump({'encodings': known_encodings, 'names': known_names}, ENCODINGS_FILE)
    print("Encodings updated and saved!")

# Function to recognize faces in real-time
def recognize_faces():
    cap = cv2.VideoCapture(0)
    print("Recognition started... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            try:
                encoding = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)[0]['embedding']
                normalized_encoding = Normalizer().fit_transform([encoding])[0]

                distances = [np.linalg.norm(normalized_encoding - enc) for enc in known_encodings]

                if distances and min(distances) < 0.4:
                    name = known_names[distances.index(min(distances))]
                    cv2.putText(frame, f"Hello {name}!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error during face recognition: {e}")
                cv2.putText(frame, "Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    while True:
        print("\nWelcome! Please choose an option:")
        print("C - Capture a new image")
        print("N - Add new face")
        print("R - Recognize faces")
        print("Q - Quit")

        choice = input("Enter your choice (C/N/R/Q): ").strip().upper()

        if choice == 'C':
            name = input("Enter the name for this capture: ").strip()
            capture_and_save_face(name)
        elif choice == 'N':
            name = input("Enter the name of the new person: ").strip()
            capture_and_save_face(name)
            encode_faces()
        elif choice == 'R':
            recognize_faces()
        elif choice == 'Q':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
