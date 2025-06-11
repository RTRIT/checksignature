from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

model = YOLO("best.pt")

#Take a PIL.Image object
#Return (image with detected bounding box) + (list of cropped signature images)
def detect_signature(image: Image.Image):
    
    image_np = np.array(image.convert("RGB"))#Convert PIL image ti RGB then to numpy array
    results = model.predict(source=image_np, save=False, conf=0.5)
    print("result: "+str(results))

    result_img = image_np.copy() #Create copy for drawing 
    cropped_images = [] #List to store the cropped signature regions E.g: (1, 3, 384, 640)
    # print("cropped signature regions: "+str(cropped_images))

    for result in results:
        for box in result.boxes.xyxy:
            print("box: "+str(box))
            x1, y1, x2, y2 = map(int, box)
            cropped = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ✅ Return both annotated image and list of crops
    return Image.fromarray(result_img), cropped_images

from PIL import Image
import numpy as np
import cv2

def enhance_signature(image: Image.Image) -> Image.Image:
    # Convert PIL to NumPy and to grayscale
    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Threshold to extract the signature (ink becomes black, background white)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Create a white background
    background = np.ones_like(image_np) * 255

    # Create blue signature (255, 0, 0) in RGB
    blue_signature = np.zeros_like(image_np)
    blue_signature[:, :] = (0, 0, 255)  # OpenCV uses BGR

    # Use the mask to blend signature with white background
    result = np.where(mask[:, :, None] == 255, blue_signature, background)

    # Convert back to PIL and return
    return Image.fromarray(result)
