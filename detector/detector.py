from ultralytics import YOLO
from image_transformer import transform_image

import cv2
import config

# Load a model
model = YOLO(config.model_path)

fake_image = None
cap = None

if config.use_fake_image:
    fake_image = cv2.imread(config.fake_image_path)
else:
    cap = cv2.VideoCapture(0)

while True:
    if fake_image is not None:
        image = fake_image
    else:
        ret, image = cap.read()

    transformed_image = transform_image(image)
    results = model(transformed_image)

    if len(results) > 0:
        first_result = results[0]

        plot = first_result.plot()
        cv2.imshow('Detected Uno Cards', plot)

        for box in first_result.boxes:
            # get coordinates of the bounding box
            # xyxy means x1 and y1 of the top left corner and x2 and y2 of the bottom right corner
            coords = box.xyxy.tolist()

            # pixel auf der mitte des rechten strichs ermitteln
    else:
        cv2.imshow('No Objects', image)

    # break the loop if the user presses escape key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
