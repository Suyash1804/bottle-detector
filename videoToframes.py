import cv2
import numpy as np
import os

video_path = os.path.join('videos', 'sunil2.mp4')
cap = cv2.VideoCapture(video_path)
cnt = 0

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

ret, first_frame = cap.read()

while cap.isOpened():
    
    ret, frame = cap.read()
    
    if ret:
      
        roi = frame[:800, :] 
        
        thresh = 100 
        if thresh * 2 < roi.shape[1]:  
            start = thresh  
            end = roi.shape[1] - thresh 
            roi = roi[:, start:end]  
        
            if roi.size > 0:
                cv2.imshow("image", roi)

                cv2.imwrite(f'frames/{cnt}.jpg', roi)
                cnt += 1
                
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            print("Threshold is too large for the current frame width. Adjust the thresh value.")

    else:
        break

cap.release()
cv2.destroyAllWindows()
