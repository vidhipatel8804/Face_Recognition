import cv2
import face_recognition
import os
import numpy as np

def load_images_and_encodings(folder_path):
    """Loads images from a folder and generates face encodings.

    Args:
        folder_path (str): Path to the folder containing known face images.

    Returns:
        tuple: A tuple of two lists:
            - known_face_encodings: List of face encodings.
            - known_face_names: List of corresponding names.
    """

    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(folder_path):
        try:
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]

            known_face_encodings.append(face_encoding)
            known_face_names.append(filename.split('.')[0])
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return known_face_encodings, known_face_names

# Specify the folder containing known faces
known_faces_folder = r"C:\Users\vidhi\OneDrive\Desktop\python\Face_Recognization\photos"  # Replace with your actual folder path

# Load images and generate encodings from the folder
known_face_encodings, known_face_names = load_images_and_encodings(known_faces_folder)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
