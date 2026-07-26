# Projeto 3 — Detecção de Máscaras Faciais (YOLO)

## 📝 Relatório do Candidato

👤 **Nome Completo: Lucas Ricardo de Lima Figueiredo**

### 1️⃣ Resumo da Abordagem

O Hiperparâmetros de fine-tuning utilizados foram:

- Épocas : 30
Tamanho da imagem : 640 x 640
Batch size : 4
cls : 0.8
cls_pw : 0.5

Por conta do desbalanceamento de classes no dataset, foram adicionados os parâmetros:
- cls(Perda de Classificação): Mede especificamente o erro do modelo na hora de adivinhar a classe correta. Se ele erra durante o treino, esse valor desce.
- cls_pw(Classification Positive Weight): Funciona como fator de ponderação. Como o banco de imagens possui muitas imagens onde as pessoas estão com máscara ou sem máscara e muito poucas com pessoas utilizando a máscara incorretamente, é possível que o modelo comece a fazer chutes errados ao encontrar essas situações raras como essa. O cls_pw da um peso maior como punição por classificar incorretamente essas classes mais raras. buscando diminuir o erro do modelo.

### 2️⃣ Bibliotecas Utilizadas

Lista das bibliotecas utilizadas:

- torch
- torchvision
- ultralytics
- opencv-python
- litert-torch

### 3️⃣ Técnica de Otimização do Modelo
A otimização do modelo foi executada através do arquivo `optimize_model.py`, a partir do arquivo `model.pt` gerado pelo `train_model.py`. A função utilizada foi a model.export() que converte a arquitetura do modelo. Os parâmetros usados em model.export() foram:

- **format = "litert":** Define que o formato será TensorFlow Lite (LiteRT)
- **imgsz = 640 :** Faz o modelo esperar por imagens de 640 x 640 pixels para inferência
- **int8=True :** O modelo é treinado ponto flutuante de 32 bits, ele simplifica isso em inteiros de 8 bits, deminuindo bastante o tamanho do arquivo.
- **data="dataset/data.yaml" :** Serve para o export encontrar o caminho das imagens que vão ser utilizadas para calibrar a conversão para int8.

No final ele usa esses parâmetros para criar o arquivo `model_int8.tflite` e modifica seu nome para o especificado: `model.tflite`.

### 4️⃣ Resultados Obtidos


Tamanho dos arquivos:
`model.pt`:
`model.tflite`:

o mAP50 foi de:

Informe o mAP50 (e, se possível, o mAP50-95) obtido na validação, por classe se
possível, e o tamanho dos arquivos `model.pt` e `model.tflite`.

### 5️⃣ Comentários Adicionais (Opcional)

As principais dificuldades encontradas ao desenvolver o projeto foram relacionadas as limitações de hardware de meu próprio computador. Maior parte das coisas precisei ir atrás de um PC emprestado e de inúmeras tentativas para funcionar. Por exemplo, meu `optimize_model.py` não conseguia rodar dentro do ambiente docker linux de minha máquina de jeito nenhum (Ele era sobrecarregado e a conexão encerrada), perdi bastante tempo procurando uma solução, a única saída foi utilizar o PC de um amigo próximo para rodar o Script sem dificuldades.

Aprendi bastante, tive que utilizar ferramentas que nunca tinha entrado em contato de forma muito aprofundada, como Docker e conteinerização. Pude entender mais sobre criação de um ambiente de desenvolvimento e as dificuldades quanto a compatibilidade entre as dependências do ambiente. Além disso, pricipalmente, consegui entender melhor como funciona o proesso de treinamento de uma IA e como os parâmetros impactam no desempenho dos modelos gerados durante o processo.

### 6️⃣ Exemplo de Inferência




Cole a saída do terminal ao rodar `run_inference.py` (número de detecções por
imagem), e comente brevemente sobre o que observou ao abrir as imagens
anotadas em `runs/detect/inferencia_exemplos/predicoes/` — por exemplo, se as
caixas ficaram bem localizadas, se houve confusão entre classes, ou se a
classe minoritária (`mask_weared_incorrect`) teve desempenho visivelmente pior.

---

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).
