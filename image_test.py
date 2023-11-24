import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

height, width, channels = frame.shape

trapeze_margin = 500

src_points = [
    [trapeze_margin, 0],
    [0, height],
    [width, height],
    [width - trapeze_margin, 0]
]

for point in src_points:
    cv2.circle(frame, (point[0], point[1]), 40, (0, 255, 0), -1)

cv2.imshow('Webcam with points', frame)
cv2.waitKey(0)

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

resized_image = cv2.resize(transformed_image, (int(width * 0.6), height))

cv2.imshow('Transformed', resized_image)

# cv2.imwrite('capture.jpg', frame)

cv2.waitKey(0)
