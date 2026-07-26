import os
import shutil
from ultralytics import YOLO


model = YOLO("model.pt")


print("Iniciando a otimização para TFLite... Esta versão será bem mais rápida!")
exported_file_path = model.export(
    format="litert", 
    imgsz=640
)

if os.path.exists(exported_file_path):
    shutil.copy(exported_file_path, "model.tflite")
    print("Sucesso! O arquivo model.tflite foi gerado na raiz do projeto.")
else:
    print("Erro: Não foi possível localizar o arquivo exportado.")