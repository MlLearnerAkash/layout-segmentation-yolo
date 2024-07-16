import json
import tqdm
from pathlib import Path

def relabelling(root_directory: Path= "./datasets/COCO"):
    for folder in ["train","val","test"]:
        print(f"Working on {folder}...")
        with open(root_directory + "/" +"COCO"+ "/" +f"{folder}.json") as f:
            big_json = json.load(f)

        for annotation in tqdm.tqdm(big_json["annotations"]) :
            category_id = annotation["category_id"]
            # print("The category id:", category_id)
            if category_id in [1, 2, 4, 5, 6, 8, 10, 11]:
                annotation["category_id"]=1
            if category_id==3:
                annotation["category_id"]=2
            if category_id==9:
                # print("Category id: ", 9)
                annotation["category_id"]=3
                # print(annotation)
                # print("Changed to annotation is: ", 3)
            if category_id==7:
                annotation["category_id"]=4
        with open(root_directory+ "/"+ "COCO"+"/"+f"{folder}_modified.json", "a") as f:
            json.dump(big_json, f, indent=4)
            # print(annotation)

if __name__=="__main__":
    relabelling("/home/akash/ws/layout-segmentation-yolo/datasets")