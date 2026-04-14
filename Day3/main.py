import cv2
from utils import get_limits
from PIL import Image

yello_color = [0, 255, 255] # yellow in BGR ColorSpace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # Convert BGR ColorSpace into RGB ColorSpace

    lowerLimit, upperLimit = get_limits(color=yello_color)

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)


    cv2.imshow("Frame ", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

