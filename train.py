import typer
from ultralytics import YOLO

def main(
    base_model: str,
    datasets: str = "./datasets/data.yaml",
    epochs: int = 500,
    imgsz: int = 1024,
    batch: int = 16,
    dropout: float = 0.0,
    resume: bool = True,
    device = "0",
    name: str= "background-finetune",
):
    try:
        from clearml import Task

        Task.init(
            project_name="yolo-doclaynet-background-230724",
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
