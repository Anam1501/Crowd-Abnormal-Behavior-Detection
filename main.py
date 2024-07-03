import cv2
import numpy as np
import os
from keras.models import load_model
from collections import deque
from twilio.rest import Client


def send_sms_alert():
    # Twilio credentials
    TWILIO_ACCOUNT_SID = 'TWILIO_ACCOUNT_SID'
    TWILIO_AUTH_TOKEN = 'TWILIO_AUTH_TOKEN'
    TWILIO_PHONE_NUMBER = '+12192717511'
    RECIPIENT_PHONE_NUMBER = '+RECIPIENT_PHONE_NUMBER'

    # Create a Twilio client
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    # Send SMS alert
    message = client.messages.create(
        body="Weapon or violence detected in the video stream!",
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )
    print("SMS Alert Sent!")

# Load YOLO model
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# Load violence detection model
def load_violence_detection_model():
    print("Loading violence detection model ...")
    model_path = 'modelnew.h5'  # Update the path to your model
    return load_model(model_path)


def detect_objects(image, confidence_threshold=0.5):
    if image is None:
        return [], [], []

    height, width, channels = image.shape

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

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
            if confidence > confidence_threshold:  # Adjust this threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indexes, boxes, class_ids


def main():
    cap = cv2.VideoCapture("violence.mp4")  # Update the path to your video file

    model = load_violence_detection_model()
    Q = deque(maxlen=128)

    while True:
        _, img = cap.read()

        indexes, boxes, class_ids = detect_objects(img, confidence_threshold=0.7)  # Adjust this threshold

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                send_sms_alert()

        # Violence detection
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame.reshape(128, 128, 3) / 255

        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)
        results = np.array(Q).mean(axis=0)
        violence_prob = results[0]

        text_color = (0, 255, 0)  # default: green
        if violence_prob > 0.50:
            text_color = (0, 0, 255)  # red
            send_sms_alert()
        text = "Violence: {:.2f}".format(violence_prob)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (35, 100), FONT, 1.25, text_color, 3)

        cv2.imshow("Combined Output", img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

