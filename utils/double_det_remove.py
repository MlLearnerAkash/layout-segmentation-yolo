from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors
import os
import torch
import cv2
import typer
def filter_detections(predictions, confidences, labels, target_label=2):
    """
    Filter out detections inside the bounding box of a specific label.

    Args:
    - predictions (torch.Tensor): The bounding boxes (in the format [x1, y1, x2, y2]).
    - confidences (torch.Tensor): The confidence scores for each bounding box.
    - labels (torch.Tensor): The labels for each bounding box.
    - target_label (int): The label of the bounding box to use for filtering.

    Returns:
    - Filtered bounding boxes, confidences, and labels.
    """
    # Find the bounding box with the target label
    target_boxes = predictions[labels == target_label]

    if len(target_boxes) == 0:
        # No target label found, return original inputs
        return predictions, confidences, labels
    # print(labels)
    target_box = target_boxes[-1]  # Assume only one target label box

    def is_inside_box(box, target_box):
        """Check if a box is inside the target box."""
        return (
            box[0] > target_box[0] and
            box[1] > target_box[1] and
            box[2] < target_box[2] and
            box[3] < target_box[3]
        )

    # Filter out boxes inside the target box
    filtered_indices = [
        i for i, box in enumerate(predictions)
        if not is_inside_box(box, target_box)
    ]

    filtered_predictions = predictions[filtered_indices]
    filtered_confidences = confidences[filtered_indices]
    filtered_labels = labels[filtered_indices]

    return filtered_predictions, filtered_confidences, filtered_labels

def main(model: str, image: str, line_width: int = 2, font_size: int = 2):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    model = YOLO(model)
    result = model.predict([img], conf = 0.4)[0]
    height = result.orig_shape[0]
    width = result.orig_shape[1]
    
    predictions = result.boxes.xyxyn
    confidences = result.boxes.conf
    labels = result.boxes.cls
    filtered_predictions, filtered_confidences, filtered_labels = filter_detections(predictions, confidences, labels)
    colors = Colors()
    annotator = Annotator(img, line_width=line_width, font_size=font_size)
    for label, box in zip(filtered_labels.tolist(), filtered_predictions.tolist()):
        label = int(label)
        annotator.box_label(
            [box[0] * width, box[1] * height, box[2] * width, box[3] * height],
            result.names[label],
            color=colors(label, bgr=True),
        )
    annotator.save(
        os.path.join(os.path.dirname(image), "annotated-" + os.path.basename(image))
    )

# Example usage
if __name__ == "__main__":
    # Load a model
    # model = YOLO("/home/akash/ws/layout-segmentation-yolo/runs/detect/train3/weights/best.pt")  # pretrained YOLOv8n model

    # # Run batched inference on a list of images
    # results = model(["/home/akash/ws/layout-segmentation-yolo/datasets/images/test/4400.png"])  # return a list of Results objects

    # print("Confidences for each cropped image:", results[0].boxes.conf)
    # print("bpounding-boxes for each cropped image:", results[0].boxes.xywhn)
    # print("classes for each cropped image:", results[0].boxes.cls)


    # # Dummy data (replace with actual data)
    # predictions = results[0].boxes.xywhn
    # confidences = results[0].boxes.conf
    # labels = results[0].boxes.cls

    # filtered_predictions, filtered_confidences, filtered_labels = filter_detections(predictions, confidences, labels)
    
    # print("Filtered Predictions:", filtered_predictions)
    # print("Filtered Confidences:", filtered_confidences)
    # print("Filtered Labels:", filtered_labels)

    typer.run(main)



