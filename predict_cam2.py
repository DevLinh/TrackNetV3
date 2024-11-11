import os
import cv2
import argparse
import onnxruntime
import numpy as np
import time
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, default='model.onnx')
parser.add_argument('--num_frame', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--save_dir', type=str, default='pred_result')
args = parser.parse_args()

model_file = args.model_file
num_frame = args.num_frame  # Keep as 3 since the model requires 3 consecutive frames
batch_size = args.batch_size
save_dir = args.save_dir

# Create output directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize ONNX model with CUDA provider
ort_session = onnxruntime.InferenceSession(model_file, providers=['CUDAExecutionProvider'])

# Write CSV file header
out_csv_file = f'{save_dir}/webcam_output.csv'
f = open(out_csv_file, 'w')
f.write('Frame,Visibility,X,Y\n')

# Webcam capture setup
cap = cv2.VideoCapture(0)  # '0' is the index of the default webcam

# Set resolution and frame rate
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 24)

# Verify settings
fps = cap.get(cv2.CAP_PROP_FPS)
frame_duration_ms = int(1000 / fps) if fps != 0 else 42  # Fallback to ~24fps if unsupported

# Get webcam dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Unable to access the webcam.")
    cap.release()
    f.close()
    exit()

# Set smaller display dimensions
display_width = 640
display_height = 360
h, w, _ = frame.shape
ratio = h / HEIGHT
frame_count = 0
success = True

# Variables to calculate average FPS
start_time = time.time()
processed_frames = 0

print("Press 'q' to exit.")

# Initialize a sliding window frame queue for processing
frame_queue = []

while success:
    # Read frame from the webcam
    success, frame = cap.read()
    if not success:
        print("Error: Unable to capture frame from webcam.")
        break

    # Resize frame to lower resolution for faster processing
    frame_resized = cv2.resize(frame, (display_width, display_height))

    # Add the frame to the queue
    frame_queue.append(frame_resized)

    # Display the first two frames as a real-time feed
    if len(frame_queue) == 1 or len(frame_queue) == 2:
        cv2.imshow("Processed Frame", frame_resized)

    # Once we have 3 frames, run inference
    if len(frame_queue) == num_frame:
        # Prepare input data for ONNX model
        x = get_frame_unit(frame_queue, num_frame)  # Shape (batch_size, num_frame*3, HEIGHT, WIDTH)
        input_name = ort_session.get_inputs()[0].name
        x_numpy = x.cpu().numpy()  # Convert to numpy array for ONNX
        y_pred = ort_session.run(None, {input_name: x_numpy})[0]  # Run inference with ONNX

        # Post-process predictions
        h_pred = (y_pred > 0.5).astype(np.uint8) * 255  # Thresholding
        h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)

        # Process the third frame in the batch (latest frame in queue)
        img = frame_queue[-1].copy()
        cx_pred, cy_pred = get_object_center(h_pred[-1])
        cx_pred, cy_pred = int(ratio * cx_pred), int(ratio * cy_pred)
        vis = 1 if cx_pred > 0 and cy_pred > 0 else 0
        # Write prediction result to CSV
        f.write(f'{frame_count},{vis},{cx_pred},{cy_pred}\n')

        # Draw the predicted point on the frame
        if cx_pred != 0 or cy_pred != 0:
            cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)

        # Display the processed frame in a separate window
        img_small = cv2.resize(img, (display_width, display_height))
        cv2.imshow("Processed Frame", img_small)

        # Update processed frames count for average FPS calculation
        processed_frames += 1

        # Clear the first frame in the queue for the next batch (maintains a sliding window)
        frame_queue.pop(0)

    # Check for 'q' key to exit
    if cv2.waitKey(frame_duration_ms) & 0xFF == ord('q'):
        success = False

    frame_count += 1

# Clean up
cap.release()
f.close()
cv2.destroyAllWindows()

# Calculate and display average FPS
end_time = time.time()
average_fps = processed_frames / (end_time - start_time)
print(f'Average processed frames per second: {average_fps:.2f}')
print('Done.')
