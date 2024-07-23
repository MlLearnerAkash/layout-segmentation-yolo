import typer
from ultralytics import YOLO

def main(
    base_model: str,
    datasets: str = "./datasets/data.yaml",
    conf: float = 0.75,
    imgsz: int = 1024,
    batch: int = 16,
    plots: bool = True,
    save_json: bool = False,
    device = "0",
    iou: float= 0.5,
):
    try:
        from clearml import Task

        Task.init(
            project_name="yolo-doclaynet-11-class-val",
            task_name=f"imgsz-{imgsz}-conf-{conf}-iou-{iou}",
        )
    except ImportError:
        print("clearml not installed")

    model = YOLO(base_model)
    results = model.val(
        data=datasets,
        conf=conf,
        imgsz=imgsz,
        batch=batch,
        plots=plots,
        save_json=save_json,
        device = device,
        iou = iou

    )
    print(results)


if __name__ == "__main__":
    typer.run(main)
