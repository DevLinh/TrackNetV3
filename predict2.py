import os
import cv2
import argparse
import onnxruntime
import numpy as np
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str)
parser.add_argument('--model_file', type=str, default='model.onnx')
parser.add_argument('--num_frame', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--save_dir', type=str, default='pred_result')
args = parser.parse_args()

video_file = args.video_file
model_file = args.model_file
num_frame = args.num_frame
batch_size = args.batch_size
save_dir = args.save_dir

video_name = video_file.split('/')[-1][:-4]
video_format = video_file.split('/')[-1][-3:]
out_video_file = f'{save_dir}/{video_name}_pred.{video_format}'
out_csv_file = f'{save_dir}/{video_name}_ball.csv'

# Create output directory if not exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load ONNX model with CUDA provider
ort_session = onnxruntime.InferenceSession(model_file, providers=['CUDAExecutionProvider'])

# Video output configuration
fourcc = cv2.VideoWriter_fourcc(*'DIVX') if video_format == 'avi' else cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(out_video_file, fourcc, int(cv2.VideoCapture(video_file).get(cv2.CAP_PROP_FPS)),
#                       (int(cv2.VideoCapture(video_file).get(cv2.CAP_PROP_FRAME_WIDTH)),
#                        int(cv2.VideoCapture(video_file).get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Write csv file header
f = open(out_csv_file, 'w')
f.write('Frame,Visibility,X,Y\n')

# Video capture setup
cap = cv2.VideoCapture(video_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ratio = h / HEIGHT
frame_count = 0
success = True

# Initialize tqdm progress bar
progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

while success:
    # Sample frames to form input sequence
    frame_queue = []
    for _ in range(num_frame * batch_size):
        success, frame = cap.read()
        if not success:
            break
        frame_queue.append(frame)
        frame_count += 1

    if not frame_queue:
        break
    
    # Handle incomplete mini-batch
    if len(frame_queue) % num_frame != 0:
        continue

    # Prepare input data for ONNX model
    x = get_frame_unit(frame_queue, num_frame)  # Shape (batch_size, num_frame*3, HEIGHT, WIDTH)
    input_name = ort_session.get_inputs()[0].name
    x_numpy = x.cpu().numpy()  # Convert to numpy array for ONNX
    y_pred = ort_session.run(None, {input_name: x_numpy})[0]  # Run inference with ONNX

    # Post-process predictions
    h_pred = (y_pred > 0.5).astype(np.uint8) * 255  # Thresholding
    h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)

    # Process each predicted frame in the batch
    for i in range(h_pred.shape[0]):
        img = frame_queue[i].copy()
        cx_pred, cy_pred = get_object_center(h_pred[i])
        cx_pred, cy_pred = int(ratio * cx_pred), int(ratio * cy_pred)
        vis = 1 if cx_pred > 0 and cy_pred > 0 else 0
        # Write prediction result to CSV
        f.write(f'{frame_count - (num_frame * batch_size) + i},{vis},{cx_pred},{cy_pred}\n')

        # Draw the predicted point on the frame
        if cx_pred != 0 or cy_pred != 0:
            cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)

        # Display and write the frame
        cv2.imshow("Prediction", img)
        # cv2.waitKey(1)
        # Control the frame rate with cv2.waitKey()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            success = False
            break
        # out.write(img)

    progress_bar.update(len(frame_queue))

# Clean up
progress_bar.close()
cap.release()
# out.release()
f.close()
cv2.destroyAllWindows()
print('Done.')
