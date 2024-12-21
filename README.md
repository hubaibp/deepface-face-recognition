create Environment 

install required Packages

start with N capture New faces

register with your name

(face_data.pkl will generate automatically)is file created using the joblib library to store the facial encoding data. 
It acts as a database for your face recognition application, storing the numerical representations (embeddings) of faces and their associated names.



Technology	          |    Purpose
----------------------|-----------------------------------------------
OpenCV	              |    Webcam input, face detection (Haar Cascade).
DeepFace (Facenet)	  |   Face encoding using pre-trained CNN.
Scikit-learn          |  	Normalization of face embeddings.
NumPy	                |    Distance computation and embedding handling.
Joblib	              |   Serialization (save and load face data).

Uses DeepFace to encode faces.
Saves and loads face encodings using joblib, allowing persistent storage for scalability.
Uses OpenCV's Haar Cascade for face detection (haarcascade_frontalface_default.xml).
Matches real-time faces against embeddings stored in a pkl file.
Persistent data ensures face encodings remain intact across sessions.
High accuracy and robustness due to deep learning models.
Persistent storage of face encodings for scalability.
