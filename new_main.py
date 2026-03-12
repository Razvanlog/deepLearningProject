import numpy as np
import cv2
import pyvirtualcam
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if width == 0 or height == 0:
    width, height = 1280, 720
fps = 30 if fps == 0 else fps

bg_image = cv2.imread("C:\\Users\\Razvan\\Pictures\\aura_bg.png")

if bg_image is None:
    print("Failed to load image. Using green screen.")
    bg_image = np.zeros((height, width, 3), dtype=np.uint8)
    bg_image[:] = (0, 255, 0)
else:
    bg_image = cv2.resize(bg_image, (width, height))

bg_image_rgb = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

with pyvirtualcam.Camera(width=width, height=height, fps=fps, backend='obs') as cam:
    print(f'Using virtual camera: {cam.device}')
    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = model(frame, classes=[0], verbose=False)
        
        if results and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = cv2.resize(mask, (width, height))
            mask_3d = np.stack((mask,) * 3, axis=-1)
            
            output_image = np.where(mask_3d > 0.1, image_rgb, bg_image_rgb)
        else:
            output_image = bg_image_rgb

        cam.send(output_image)
        cam.sleep_until_next_frame()
        
        output_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Local AI Preview', output_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
