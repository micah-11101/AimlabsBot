import cv2
import numpy as np
import win32api
import win32con
import time
import keyboard
from PIL import ImageGrab
from threading import Thread

# Load YOLOv4 model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Global variable to control the main loop
running = True

# Sensitivity for mouse movement
sensitivity = 1.15

fov = 0.7

def capture_screen():
    screenshot = np.array(ImageGrab.grab())
    return cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

def find_target(screen):
    height, width, _ = screen.shape
    blob = cv2.dnn.blobFromImage(screen, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0 is the class ID for person
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.1)
    if len(indexes) > 0:
        i = indexes[0]
        box = boxes[i]
        x, y, w, h = box
        return (int(x + w/2), int(y + h/2))
    return None

def mouse_move(dx, dy):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

def mouse_click(dx,dy):
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, int(dx), int(dy), 0, 0)
    time.sleep(0.0001)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, int(dx), int(dy), 0, 0)

def check_exit():
    global running
    while running:
        if keyboard.is_pressed("capslock"):
            print("Ending script")
            running = False
        time.sleep(0.1)

def main():
    global running
    
    # Start the exit check thread
    exit_thread = Thread(target=check_exit)
    exit_thread.start()
    
    time.sleep(3)  # Initial delay
    
    while running:
        time.sleep(0.01)
        screen = capture_screen()
        target_location = find_target(screen)
        
        if target_location:
            target_x, target_y = target_location
            cursor_x, cursor_y = win32api.GetCursorPos()
            
            # Calculate the difference between target and cursor position
            dx = (target_x - cursor_x) * sensitivity
            dy = (target_y - cursor_y) * sensitivity
            adj_x = round((dx * fov))
            adj_y = round((dy * fov))
            # Move the mouse relative to its current position
            mouse_move(dx, dy)
            time.sleep(0.01)
            mouse_click(dx,dy)
            print(f"Target found. Mouse moved by: ({dx}, {dy})")
        
          # Reduced sleep time for faster response
    
    exit_thread.join()

if __name__ == "__main__":
    main()
