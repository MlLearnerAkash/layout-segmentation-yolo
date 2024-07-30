import typer
from ultralytics import YOLO

def main(
    base_model: str,
    datasets: str = "/home/akash/ws/dataset/datasets/publaynet/JSON2YOLO/new_dir/dataset.yaml",
    epochs: int = 150,
    imgsz: int = 1024,
    batch: int = 16,
    dropout: float = 0.0,
    resume: bool = False,
    device = "0",
    name: str= "publaynet-finetune",
):
    try:
        from clearml import Task

        Task.init(
            project_name="yolo-publaynet-30072024",
            task_name=f"base-model-{base_model}-epochs-{epochs}-imgsz-{imgsz}-batch-{batch}",
        )
    except ImportError:
        print("clearml not installed")

    model = YOLO(base_model)
    results = model.train(
        data=datasets,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        dropout=dropout,
        resume=resume,
        device = device,
        name= name,

    )
    print(results)


if __name__ == "__main__":
    typer.run(main)
