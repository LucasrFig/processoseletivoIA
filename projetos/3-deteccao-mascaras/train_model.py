import shutil

from ultralytics import YOLO


# insira seu código aqui

model = YOLO("yolo11n.pt")

print("Iniciando o treinamento. Isso pode levar algum tempo na CPU...")
results = model.train(
    data="dataset/data.yaml",
    epochs=1,
    imgsz=640, 
    batch=4,   
    device="cpu"
)

best_weights_path = results.save_dir / "weights" / "best.pt"
shutil.copy(best_weights_path, "model.pt")
print("O arquivo model.pt foi gerado.")
