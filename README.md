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
