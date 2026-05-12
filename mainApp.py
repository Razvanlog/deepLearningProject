import customtkinter as ctk
from tkinter import filedialog
import numpy as np
import cv2
import pyvirtualcam
from ultralytics import YOLO
import threading
import sys

ctk.set_appearance_mode("dark")

ctk.set_default_color_theme("dark-blue")


class app(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("app")
        self.geometry("500*400")

        self.bg_image_path = None
        self.bg_image_rgb = None
        self.runs= None

        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(self, text="Background", font= ctk.CTkFont(size=20, weight="bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(20,10))

        self.bg_btn = ctk.CTkButton(self, text="Select Background Image", command=self.select_bg_image)
        self.bg_btn.grid(row=1, column=0, padx=20, pady=10)

        self.bg_label  = ctk.CTkLabel(self, text="Default: green screen", font= ctk.CTkFont(size=14))
        self.bg_label.grid(row=2, column=0, padx=20, pady=10)

        device_label = ctk.CTkLabel(self, text="Device", font= ctk.CTkFont(size=14))
        device_label.grid(row=3, column=0, padx=20, pady=(20,10))

        self.device_var = ctk.StringVar(value="gpu (cuda)")

        self.device_dropdown = ctk.CTkOptionMenu(self, values=["gpu (cuda)", "cpu"], variable=self.device_var)

        self.device_dropdown.grid(row=4, column=0, padx=20, pady=10)


        self.start_btn = ctk.CTkButton(self,
                                       text="Start Virtual Camera",
                                       fg_color="green",
                                       command=self.toggle_camera
                                       )
        self.start_btn.grid(row=5, column=0, padx=20, pady=30)

    def select_bg_image(self):
        filepath = filedialog.askopenfilename(title="Select Background Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if filepath:
            self.bg_image_path = filepath
            self.bg_label.configure(text=f"...{filepath[-30:]}")

            img = cv2.imread(filepath)
            if img is not None:
                self.bg_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                self.bg_label.configure(text="Failed to load image. green screen background.")
                self.bg_image_rgb = None

    def toggle_camera(self):
        if not self.runs:
            self.runs = True
            self.start_btn.configure(text="Stop Camera", fg_color="red", hover_color="darkred")
            self.device_dropdown.configure(state="disabled")
            self.bg_btn.configure(state="disabled")
            
            threading.Thread(target=self.run_camera_loop, daemon=True).start()
        else:
            self.runs = False
            self.start_btn.configure(text="Start Virtual Camera", fg_color="green", hover_color="darkgreen")
            self.device_dropdown.configure(state="normal")
            self.bg_btn.configure(state="normal")


    def run_camera_loop(self):
        device_choice = '0' if self.device_var.get() == "GPU (CUDA)" else 'cpu'
        
        try:
            model = YOLO('best.pt') 
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.runs = False
            return

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Cannot open camera")
            self.runs = False
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if width == 0 or height == 0:
            width, height = 1280, 720
        fps = 30 if fps == 0 else fps

        if self.bg_image_rgb is None:
            bg = np.zeros((height, width, 3), dtype=np.uint8)
            bg[:] = (0, 255, 0)
        else:
            bg = cv2.resize(self.bg_image_rgb, (width, height))

        try:
            with pyvirtualcam.Camera(width=width, height=height, fps=fps, backend='obs') as cam:
                print(f'Virtual camera started: {cam.device}')
                
                while self.runs:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    results = model(frame, conf=0.5, classes=[0], device=device_choice, verbose=False)
                    
                    combined_mask = np.zeros((height, width), dtype=np.float32)

                    if results[0].masks is not None:
                        masks = results[0].masks.data.cpu().numpy()
                        if masks is not None and len(masks) > 0:
                            combined_mask = np.max(masks, axis=0)
                            combined_mask = cv2.resize(combined_mask, (width, height))
                            
                            combined_mask = cv2.GaussianBlur(combined_mask, (15, 15), 0)

                    alpha = np.expand_dims(combined_mask, axis=-1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    output_image = (frame_rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)

                    cam.send(output_image)
                    cam.sleep_until_next_frame()
                    
                    # Local Preview Window
                    output_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow('NeuralTech Preview', output_bgr)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.runs = False
                        break

        except Exception as e:
            print(f"Virtual Camera Error: {e}\n(Is OBS Virtual Camera installed and enabled?)")

        cap.release()
        cv2.destroyAllWindows()
        
        self.start_btn.configure(text="Start Virtual Camera", fg_color="green", hover_color="darkgreen")
        self.device_dropdown.configure(state="normal")
        self.bg_btn.configure(state="normal")


if __name__ == "__main__":
    a  = app()

    def on_closing():
        a.runs = False
        a.destroy()
        sys.exit()


    a.protocol("WM_DELETE_WINDOW", on_closing)
    a.mainloop()