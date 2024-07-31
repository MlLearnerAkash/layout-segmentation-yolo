import os.path

import cv2
import typer
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors

FILTERED_BOXES= False

def main(model: str, image: str, line_width: int = 2, font_size: int = 8):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    model = YOLO(model)
    result = model.predict(img, conf = 0.1)[0]
    # print(result.boxes.conf)
    # print(result.boxes.xyxyn[result.boxes.conf < 0.25])
    # print(result.boxes.xyxyn)
    filtered_xyxyn = result.boxes.xyxyn[result.boxes.conf < 0.25]
    # result.boxes.xyxyn = filtered_xyxyn
    height = result.orig_shape[0]
    width = result.orig_shape[1]

    if FILTERED_BOXES:
        boxes = filtered_xyxyn
    else:
        boxes = result.boxes.xyxyn
    colors = Colors()
    annotator = Annotator(img, line_width=line_width, font_size=font_size)
    for label, box in zip(result.boxes.cls.tolist(), boxes.tolist()):
        label = int(label)
        annotator.box_label(
            [box[0] * width, box[1] * height, box[2] * width, box[3] * height],
            result.names[label],
            color=colors(label, bgr=True),
        )
    annotator.save(
        os.path.join(os.path.dirname(image), "annotated-" + os.path.basename(image))
    )


if __name__ == "__main__":
    typer.run(main)
