import cv2
import numpy as np
from ultralytics import YOLO

# Load the video
vid = cv2.VideoCapture(r"D:\CODES\Projects\mounarchtech\karad2\Karad Railway Line.mp4")

# Load the model
model = YOLO("karad2/rail3.pt")

CLS_NAMES = {0: "Overhead-Electric-Poles", 1: "Railway Track"}


def railway(IMAGE):
    # Creating Dict to store necessory data
    data = {"masks": [], "score": [], "names": []}

    result = model.predict(IMAGE)

    # Storing Xy cord of polygon
    data["masks"] = result[0].masks.xy

    if len(data["masks"]) == 0:
        return None
    else:

        # DATA[names] contains actual class names which we created during training
        data["names"] = result[0].boxes.cls
        data["masks"] = result[0].masks.xy
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

        for mask, score, pname in zip(data["masks"], data["score"], data["names"]):

            Name_ = CLS_NAMES[int(pname)]
            mask = np.array(mask, np.int32)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", float(score))

            ####### Drawing polygon
            if float(score) > 0.60:
                name = Name_
                # electric pole ends at frame top thats we keep the name at lower
                if Name_ == "Overhead-Electric-Poles":
                    frame = cv2.polylines(
                        frame, [mask], isClosed=True, color=(0, 255, 255), thickness=1
                    )
                    text_x, text_y = (
                        mask[0][0],
                        mask[:, 1].max() + 30,
                    )
                    cv2.putText(
                        frame,
                        Name_,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 255, 255),
                        2,
                    )

                else:
                    frame = cv2.polylines(
                        frame, [mask], isClosed=True, color=(255, 255, 0), thickness=1
                    )
                    text_x, text_y = (
                        mask[0][0],
                        mask[:, 1].min() - 10,
                    )  # Above the polygon

                    cv2.putText(
                        frame,
                        Name_,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 0),
                        2,
                    )

            else:
                pass

    cv2.imshow("color image", frame)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break


vid.release()

cv2.destroyAllWindows()
