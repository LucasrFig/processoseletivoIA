import os
import shutil
from ultralytics import YOLO


model = YOLO("model.pt")

# Otimização do modelo model.pt
print("Iniciando a otimização para TFLite...")
exported_file_path = model.export(
    format="litert",
    imgsz=640,
    int8=True,
    data="dataset/data.yaml"
)

if os.path.exists(exported_file_path):
    shutil.copy(exported_file_path, "model.tflite")
    print("Pronto! O arquivo model.tflite foi gerado na raiz do projeto.")
else:
    print("Erro: Não foi possível localizar o arquivo exportado.")
