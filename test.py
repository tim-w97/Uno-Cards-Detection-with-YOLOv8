from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("/opt/homebrew/runs/detect/train3/weights/best.pt")


def detect_from_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        height, width, channels = frame.shape

        trapeze_margin = 500

        src_points = [
            [trapeze_margin, 0],
            [0, height],
            [width, height],
            [width - trapeze_margin, 0]
        ]

        dst_points = [
            [0, 0],
            [0, height],
            [width, height],
            [width, 0]
        ]

        matrix = cv2.getPerspectiveTransform(
            np.float32(src_points),
            np.float32(dst_points)
        )

        transformed_image = cv2.warpPerspective(frame, matrix, (width, height))

        resized_image = cv2.resize(transformed_image, (int(width * 0.4), height))

        rotated_image = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)

        results = model(rotated_image)

        if len(results) > 0:
            plot = results[0].plot()
            cv2.imshow('Uno Cards', plot)
        else:
            cv2.imshow('No Objects', rotated_image)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # If 'q' is pressed, break the loop
            print("Exiting... 'q' key pressed.")
            break
        elif key == ord('s'):  # If 's' is pressed, save the image
            cv2.imwrite('captured_image.jpg', rotated_image)
            print("Image captured and saved as captured_image.jpg")

    cap.release()
    cv2.destroyAllWindows()


detect_from_webcam()
