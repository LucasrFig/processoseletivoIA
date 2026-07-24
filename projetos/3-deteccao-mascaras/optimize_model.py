import os
import shutil
from ultralytics import YOLO
# ---------------------------------------------------------------------------
# Projeto 3 — Otimização do Modelo (Exportação para Edge)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.pt"
#   2. Exportar para TensorFlow Lite via model.export(format="tflite")
#      (a Ultralytics gera automaticamente "model.tflite" na mesma pasta)
# ---------------------------------------------------------------------------

# insira seu código aqui

# 1. Carrega o modelo que você acabou de treinar
model = YOLO("model.pt")

# 2. Exporta o modelo para o formato TFLite
print("Iniciando a otimização para TFLite...")
exported_file_path = model.export(format="tflite")

# 3. Garante que o arquivo final se chame exatamente "model.tflite" na raiz
if os.path.exists(exported_file_path):
    shutil.copy(exported_file_path, "model.tflite")
    print("O arquivo model.tflite foi gerado na raiz do projeto.")
else:
    print("Erro: Não foi possível localizar o arquivo exportado.")