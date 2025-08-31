import cv2
import os

thres = 0.45  # Threshold

cap = cv2.VideoCapture(0)  
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# Load class names
classNames = []
with open("coco.names", "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Model files
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

# Load model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Video saving setup
save_path = os.path.abspath("detected_output.mp4")
frame_width, frame_height = 800, 400   # match resize
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

while True:
    success, img = cap.read()
    if not success:
        break

    # Resize frame
    img = cv2.resize(img, (frame_width, frame_height))

    # Detect objects
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, (0,0,255), 2)
            cv2.putText(img, classNames[classId - 1].upper()
,
                        (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255,0, 0), 2)
         

    # ✅ Save processed frame (with text + rectangles)
    out.write(img)

    # Show live
    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord("a"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Video saved at: {save_path}")
