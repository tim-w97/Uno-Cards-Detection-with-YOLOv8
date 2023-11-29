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
        plot = results[0].plot()
        cv2.imshow('Uno Cards', plot)
    else:
        cv2.imshow('No Objects', image)

    # break the loop if the user presses escape key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
