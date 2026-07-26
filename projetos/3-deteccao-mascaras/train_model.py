import shutil

from ultralytics import YOLO

# insira seu código aqui

model = YOLO("yolo11n.pt")

print("Iniciando o treinamento em CPU:")
results = model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=4,
    device="cpu",
    cls=0.8,
    cls_pw=0.5
)

best_weights_path = results.save_dir / "weights" / "best.pt"
shutil.copy(best_weights_path, "model.pt")
print("O arquivo model.pt foi gerado.")
