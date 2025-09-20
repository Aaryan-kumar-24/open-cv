import cv2
import imutils
import datetime

# Load your trained gun cascade
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Open IP camera stream
camera = cv2.VideoCapture("http://192.168.240.203:4747/video")

while True:
    ret, frame = camera.read()
    if frame is None:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect guns
    guns = gun_cascade.detectMultiScale(gray, 1.3, 20, minSize=(100, 100))

    for (x, y, w, h) in guns:
        # Draw rectangle around each gun
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Put label above each rectangle
        cv2.putText(frame, "WEAPONS", (x, y - 10),  # position above rectangle
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display timestamp
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"),
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow("Security Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
