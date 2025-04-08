from collections import deque
from ultralytics import YOLO
import math
import time
import cv2
import os
import numpy as np
from threading import Thread
from sklearn.metrics import precision_score, recall_score, f1_score

# Angle calculation function
def angle_between_lines(m1, m2=1):
    if m1 != -1/m2:
        angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
        return angle
    else:
        return 90.0

# Centroid history queue
class FixedSizeQueue:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)
    
    def add(self, item):
        self.queue.append(item)
    
    def pop(self):
        self.queue.popleft()
        
    def clear(self):
        self.queue.clear()

    def get_queue(self):
        return self.queue
    
    def __len__(self):
        return len(self.queue)

# Load model and video
model_path = os.path.join('runs', 'detect', 'train17', 'weights', 'best.pt')
model = YOLO(model_path)

video_path = os.path.join('videos', 'swing1.mp4')
cap = cv2.VideoCapture(video_path)

# Efficiency Measurement Variables
start_time = time.time()
frame_count = 0
fps_update_interval = 10
track_interval = 3  # Reduced for better recall

prev_frame_time = time.time()
new_frame_time = prev_frame_time
fps_text = "FPS: 0"
centroid_history = FixedSizeQueue(25)  # Increased for smoother tracking
start_time = time.time()
interval = 0.6
paused = False
angle = 0

# Performance Metrics Variables
ground_truth = []
predictions = []

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % fps_update_interval == 0:
        new_frame_time = time.time()
        fps = int(fps_update_interval / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        fps_text = f'FPS: {fps}'

    if frame_count % track_interval == 0:
        results = model.track(frame, persist=True, conf=0.28, iou=0.45, verbose=False)  # Improved conf + NMS added
        boxes = results[0].boxes
        box = boxes.xyxy
        rows, cols = box.shape

        # Collect performance metrics data
        predictions.append(len(box) > 0)  
        ground_truth.append(True)  

        if len(box) != 0:
            for i in range(rows):
                x1, y1, x2, y2 = box[i].tolist()
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                centroid_history.add((centroid_x, centroid_y))
                cv2.circle(frame, (centroid_x, centroid_y), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    if len(centroid_history) > 1:
        cv2.polylines(frame, [np.array(centroid_history.get_queue(), dtype=np.int32)],
                      isClosed=False, color=(255, 0, 0), thickness=4)

    if len(centroid_history) > 1:
        centroid_list = list(centroid_history.get_queue())
        x_diff = centroid_list[-1][0] - centroid_list[-2][0]
        y_diff = centroid_list[-1][1] - centroid_list[-2][1]
        if x_diff != 0:
            m1 = y_diff / x_diff
            if m1 == 1:
                angle = 90
            elif m1 != 0:
                angle = 90 - angle_between_lines(m1)
            if angle >= 45:
                print("Ball bounced")

        future_positions = [centroid_list[-1]]
        for i in range(1, 5):
            future_positions.append((
                centroid_list[-1][0] + x_diff * i,
                centroid_list[-1][1] + y_diff * i
            ))
        for i in range(1, len(future_positions)):
            cv2.line(frame, future_positions[i-1], future_positions[i], (0, 255, 0), 4)
            cv2.circle(frame, future_positions[i], radius=3, color=(0, 255, 0), thickness=-1)

    text = "Angle: {:.2f} degrees".format(angle)
    cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.putText(frame, fps_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    frame_resized = cv2.resize(frame, (1000, 600))
    cv2.imshow('frame', frame_resized)

    time.sleep(max(0, 1/30 - (time.time() - new_frame_time)))

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord(' '):
        paused = not paused
        while paused:
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                paused = not paused
            elif key == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

# Efficiency Calculation
end_time = time.time()
total_time = end_time - start_time
print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
print(f"Average FPS: {frame_count / total_time:.2f}")

# Performance Metrics Calculation
precision = precision_score(ground_truth, predictions)
recall = recall_score(ground_truth, predictions)
f1 = f1_score(ground_truth, predictions)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
