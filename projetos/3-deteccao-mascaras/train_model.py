import shutil

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Detecção de Máscaras Faciais (Fine-tuning do YOLO11n)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo pré-treinado YOLO11n: YOLO("yolo11n.pt")
#      (única exceção à regra de "sem modelos pré-treinados" do processo seletivo)
#   2. Fazer fine-tuning em dataset/data.yaml, em CPU (device="cpu"),
#      com um número de épocas modesto (ex: 15-30)
#   3. Copiar os pesos resultantes (results.save_dir / "weights" / "best.pt")
#      para "model.pt", na raiz desta pasta
# ---------------------------------------------------------------------------

# insira seu código aqui

# 1. Carregar o modelo pré-treinado YOLO11n
model = YOLO("yolo11n.pt")

# 2. Fazer fine-tuning em dataset/data.yaml, em CPU
print("Iniciando o treinamento. Isso pode levar algum tempo na CPU...")
results = model.train(
    data="dataset/data.yaml",
    epochs=5, 
    imgsz=640, 
    batch=4,   
    device="cpu"
)

# 3. Copiar os pesos resultantes para "model.pt" na raiz da pasta
best_weights_path = results.save_dir / "weights" / "best.pt"
shutil.copy(best_weights_path, "model.pt")
print("O arquivo model.pt foi gerado.")
