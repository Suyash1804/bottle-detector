from ultralytics import YOLO
import cv2
import numpy as np
import logging

# Suppress all logging output
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

# Load a model
model = YOLO("yolov5su.pt")


# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

img = cv2.imread("./bus.jpg")
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

def plot_detections(image, detections):
    for detection in detections:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        
        # Extract class label and confidence
        class_id = int(detection.cls[0])
        confidence = detection.conf[0]
        label = f"{model.names[class_id]}: {confidence:.2f}"
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put label above the bounding box
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 5)

    return image

def colored_bw_mask(background, bw_image, color=(0,0,255), alpha=0.5):
    # Ensure background is color image
    if len(background.shape) < 3:
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    
    # Create mask
    _, mask = cv2.threshold(bw_image, 127, 255, cv2.THRESH_BINARY)
    
    # Resize if needed
    mask = cv2.resize(mask, (background.shape[1], background.shape[0]))
    
    # Create colored overlay
    colored_overlay = np.zeros_like(background)
    colored_overlay[mask == 255] = color
    
    # Blend images
    result = cv2.addWeighted(background, 1, colored_overlay, alpha, 0)
    
    return result

def gamma_trans(img, gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)

while True:
    _, frame = cam.read()
    frame = gamma_trans(frame, 1)

    res = model(frame)
    for r in res:
        if r is not None:
            for i, cls in enumerate(r.boxes.cls):
                if(cls == 65 or cls == 39 or cls == 67):
                    box = r.boxes.xyxy[i].numpy()
                    bbox = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,255), 5)
                    cv2.putText(frame, "bottle", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5)


    cv2.imshow("bbox", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
