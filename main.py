import cv2
import numpy as np
from ultralytics import YOLO

# Load the video
vid = cv2.VideoCapture(r"D:\CODES\Projects\mounarchtech\karad2\Karad Railway Line.mp4")

# Load the model
model = YOLO("karad2/rail1.pt")

CLS_NAMES = {0: "Animal", 1: "Human", 2: "Vehicle"}


def railway(IMAGE):
    # Creating Dict to store necessory data
    data = {"rect": [], "score": [], "names": []}

    result = model.predict(IMAGE)

    # Storing Xy cord of rectangle
    data["rect"] = result[0].boxes.xyxy

    if len(data["rect"]) == 0:
        return None
    else:

        # DATA[names] contains actual class names which we created during training
        data["names"] = result[0].boxes.cls
        data["rect"] = result[0].boxes.xyxy
        data["score"] = result[0].boxes.conf

        print("###############")
        print(data)
        print("##############")

        return data


while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # model prediction

    data = railway(IMAGE=frame)
    if data is None:
        print("_--_______________no data found_________________--_")

    else:
        print("----WORKING----")

        for sublist, score, pname in zip(data["rect"], data["score"], data["names"]):
            x1, y1, x2, y2 = (
                int(sublist[0]),
                int(sublist[1]),
                int(sublist[2]),
                int(sublist[3]),
            )

            Name_ = CLS_NAMES[int(pname)]
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", float(score))

            ####### Drawing rectangel
            if float(score) > 0.60:

                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                print("WE got the cordinates")
                # giving  text
                name = Name_
                cv2.putText(
                    frame,
                    text=name,
                    org=(x1, y1 - 10),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

            else:
                pass

    cv2.imshow("color image", frame)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break


vid.release()

cv2.destroyAllWindows()
