import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
from collections import deque
from ultralytics import YOLO
import spacy

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Load YOLO model for object detection
object_detection_model = YOLO("yolov8x.pt")

# Load LSTM model for human activity recognition
activity_recognition_model = load_model('C:/Users/HP/PycharmProjects/OD/HAR3333.h5')

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Define labels for objects and activities
object_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bow', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

activity_labels = ['Carrying', 'Checking the time on a watch', 'Drinking', 'Eating', 'Reading', 'Standing', 'Talking through phone', 'Using a computer', 'Walking', 'Writing']

# Initialize deque to store recent frames5+------------------------------------------------++++++
buffer_size = 33
frame_buffer = deque(maxlen=buffer_size)

# Open the video file
video_file = 'C:/Users/HP/Downloads/HAR Dataset/HAR VIDEOS/Using a Computer/3.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_file)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create an output video write2
#

# +r
output_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MP4 codec
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Initialize variables to store previous activity and objects
prev_activity = ""
prev_objects = []

# Initialize font and text position for overlaying text on video frames
font = cv2.FONT_HERSHEY_SIMPLEX
text_position = (10, 50)

# Process video to detect poses, objects, and recognize activities
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame to detect poses
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frame_rgb)

    if results_pose.pose_landmarks is not None:
        # Extract pose landmarks
        pose_landmarks = [[lm.x, lm.y, lm.z] for lm in results_pose.pose_landmarks.landmark]
        frame_buffer.append(pose_landmarks)

        # If the window is filled, predict activity
        if len(frame_buffer) == buffer_size:
            # Preprocess the frames
            processed_frames = np.array(frame_buffer)  # Convert to numpy array
            processed_frames = processed_frames.reshape(-1, 33, 3)  # Reshape to match model input shape

            # Predict activity
            predictions_activity = activity_recognition_model.predict(processed_frames)
            average_confidence = np.mean(predictions_activity, axis=0)

            # Get predicted activity
            predicted_activity = activity_labels[np.argmax(average_confidence)]
            confidence = average_confidence[np.argmax(average_confidence)]

            # Object detection
            results_object = object_detection_model.predict(frame, show=False)

            # Draw the detected objects and activity on the frame
            output_sentence = f"The person is {predicted_activity.lower()}"

            # Display combined sentence on the frame
            cv2.putText(frame, output_sentence, text_position, font, 1, (0, 255, 0), 2)
            print(output_sentence)
        # Write frame to output video
        out.write(frame)

    # Display the frame with detected objects and activity
    cv2.imshow('ODHAR', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Function to load pose data
def load_pose_data(data_path):
    # Your implementation here

# Load the test data
    data_path = 'C:/Users/HP/Downloads/HAR Dataset/HAR POSES'
    X_test, y_test = load_pose_data(data_path)

    # Load the trained model
    activity_recognition_model = load_model('C:/Users/HP/PycharmProjects/OD/HAR3333.h5')

    # Predict on test data
    y_pred = activity_recognition_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Generate classification report
    report = classification_report(y_test, y_pred_classes, output_dict=True)

    # Extract precision, recall, and F1-score
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1_score = report['macro avg']['f1-score']

    # Plotting
    metrics = ['Precision', 'Recall', 'F1-score']
    scores = [precision, recall, f1_score]

    plt.figure(figsize=(8, 6))
    plt.bar(metrics, scores, color=['blue', 'orange', 'green'])
    plt.title('Metrics for Human Activity Recognition Model')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.ylim(0, 1)  # Limit y-axis to [0, 1]
    plt.grid(axis='y')
    plt.show()
