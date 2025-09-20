import cv2
import numpy as np

thres = 0.45  # Threshold to detect object
nms_threshold = 0.2

# Use IP webcam stream (make sure the IP is correct and phone is connected)
cap = cv2.VideoCapture("http://10.160.198.251:4747/video")
# cap.set(3,1280)
# cap.set(4,720)
# cap.set(10,150)

# Load class names
classFile = 'coco.names'
objects = [ "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "street sign", "stop sign", 
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", 
    "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", 
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", 
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
    "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", 
    "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
    "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush", "hair brush"
]


with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect objects
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    # Draw results safely
    if len(indices) > 0:
        for i in np.array(indices).flatten():  # Works across OpenCV versions
            box = bbox[i]
            x, y, w, h = box
            if classNames[classIds[i] - 1] in objects:

                cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                label = f"{classNames[classIds[i] - 1].upper()} {confs[i]:.2f}"

                cv2.putText(img, "weapon", (x + 10, y + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", img)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
