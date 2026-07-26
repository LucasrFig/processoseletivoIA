# Projeto 3 — Detecção de Máscaras Faciais (YOLO)

## 📝 Relatório do Candidato

👤 **Nome Completo: Lucas Ricardo de Lima Figueiredo**

### 1️⃣ Resumo da Abordagem

O Hiperparâmetros de fine-tuning utilizados foram:

- Épocas : 30
- Tamanho da imagem : 640 x 640
- Batch size : 4
- cls : 0.8
- cls_pw : 0.5

Por conta do desbalanceamento de classes no dataset, foram adicionados os parâmetros:
- **cls(Perda de Classificação):** Mede especificamente o erro do modelo na hora de adivinhar a classe correta. O valor do cls desce quando o erro do treinamento diminui. O trabalho dele é punir o modelo (aumentar o valor de perda) toda vez que ele encontra um rosto, mas erra a classe.
- **cls_pw(Classification Positive Weight):** Funciona como fator de ponderação. O dataset possui muitas imagens onde as pessoas estão com máscara ou sem máscara, porém ele possui poucas imagens com pessoas utilizando a máscara incorretamente, é possível que o modelo comece a fazer chutes incorretos ao encontrar situações raras como essa. O cls_pw dá um peso maior como punição por classificar incorretamente essas classes mais raras, buscando diminuir o erro do modelo.

### 2️⃣ Bibliotecas Utilizadas

Lista das bibliotecas utilizadas no ambiente local:

* `torch==2.12.1`
* `torchvision==0.27.1`
* `ultralytics==8.4.106`
* `litert-torch==0.9.1`
* `opencv-python==5.0.0.93`

### 3️⃣ Técnica de Otimização do Modelo
A otimização do modelo foi executada através do arquivo `optimize_model.py` a partir do arquivo `model.pt` gerado pelo `train_model.py`. A função utilizada foi a `model.export()` que converte a arquitetura do modelo. Os parâmetros usados em `model.export()` foram:

- **format = "litert":** Define que o formato será TensorFlow Lite (LiteRT).
- **imgsz = 640 :** Faz o modelo esperar por imagens de 640 x 640 pixels para inferência.
- **int8 = True :** O modelo é treinado ponto flutuante de 32 bits, ele simplifica o modelo em inteiros de 8 bits, diminuindo bastante o tamanho do arquivo.
- **data = "dataset/data.yaml" :** Como a opção **int8 = True** foi escolhida, serve para a função export encontrar o caminho das imagens que vão ser utilizadas para calibrar a conversão para int8.

Os últimos 2 parâmetros servem para diminuir o tamanho do modelo para sua utilização em dispositivos de borda.
No final, o `optimize_model.py` usa esses parâmetros para criar o arquivo `model_int8.tflite` e modifica seu nome para o especificado: `model.tflite`.

### 4️⃣ Resultados Obtidos
O modelo obteve os seguintes resultados, dentro do conjunto de validação, depois de 30 épocas de fine-tunning.
(Dados do arquivo results.png):

#### Resultados do modelo original(model.pt):

- mAP50 global: 82.82%
- mAP50-95 global: 57.49%

#### Resultados do modelo otimizado(model.tflite):

- mAP50 global: 73.6%
- mAP50-95 global: 43.5%

#### Comparativo de Desempenho por Classe (mAP50)

| Classe | Precisão Original (`.pt`) | Precisão Otimizada (`.tflite`) |
| :--- | :--- | :--- |
| `with_mask` | 96.6% | 86.7% |
| `without_mask` | 79.1% | 67.2% |
| `mask_weared_incorrect` | 72.8% | 66.7% |

A classe de menor desempenho foi a `mask_weared_incorrect`, o que já era esperado por conta do desbalanceamento da base de imagens. Porém, ainda teve um desempenho satisfatório devido ao parâmetros acrescentados no treinamento do modelo.

#### Tamanho dos arquivos:
- `model.pt`: 5325 KB (~5.33MB)
- `model.tflite`: 2971 KB (~2.97MB)

### 5️⃣ Comentários Adicionais (Opcional)

As principais dificuldades encontradas ao desenvolver o projeto foram relacionadas às limitações de hardware de meu próprio computador. Maior parte das coisas precisei ir atrás de um PC emprestado e de inúmeras tentativas para funcionar. Por exemplo, meu `optimize_model.py` não conseguia rodar dentro do ambiente docker linux de minha máquina de jeito nenhum (Ele era sobrecarregado e a conexão era encerrada com o contâiner), perdi bastante tempo procurando uma solução, a única saída foi utilizar o PC de um amigo para rodar o Script sem dificuldades.

Aprendi bastante, tive que utilizar ferramentas que nunca tinha entrado em contato de forma muito aprofundada, como Docker e conteinerização. Pude entender mais sobre criação de um ambiente de desenvolvimento e as dificuldades quanto a compatibilidade entre as dependências do ambiente. Além disso, principalmente, consegui entender melhor como funciona o processo de treinamento de uma IA e como os parâmetros impactam no desempenho dos modelos gerados durante o processo.

### 6️⃣ Exemplo de Inferência

#### Saídas do `run_inference.py`:
```
Projeto 3 — Inferência com model.tflite (Edge AI)
============================================================

Rodando inferência em 15 amostras usando model.tflite:

Imagem                               Detecções  Detalhes
----------------------------------------------------------------------
Loading /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/model.tflite for LiteRT inference...
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss105.jpg                         11  [11x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss107.jpg                          1  [1x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss11.jpg                          41  [1x mask_weared_incorrect, 40x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss113.jpg                          9  [8x with_mask, 1x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss12.jpg                          24  [20x with_mask, 4x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss123.jpg                          2  [2x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss124.jpg                          9  [5x without_mask, 4x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss126.jpg                          2  [2x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss128.jpg                          1  [1x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss129.jpg                          5  [2x with_mask, 3x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss130.jpg                          6  [1x mask_weared_incorrect, 3x without_mask, 2x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss135.jpg                          5  [2x with_mask, 3x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss139.jpg                         31  [1x mask_weared_incorrect, 28x with_mask, 2x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss142.jpg                          1  [1x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss143.jpg                          1  [1x with_mask]
----------------------------------------------------------------------
TOTAL                                      149

✅ Imagens anotadas salvas em: runs/detect/inferencia_exemplos/predicoes/
   (Abra essa pasta para verificar visualmente as bounding boxes preditas)
```

#### Comentários:

As Caixas estavam bem localizadas, não consegui identificar dificuldade ao identificar pessoas com máscara ou sem máscara, em todas em que avaliei ele acertou. Porém a classe minoritária (Pessoas usando a máscara incorretamente) teve um comportamento estranho, em algumas imagens a mesma pessoa recebia 2 caixas: uma dizendo que estava usando máscara e outra dizendo que estava usando incorretamente. Mas de modo geral, o modelo conseguiu identificar o rosto e classificar corretamente.

Pesquisei sobre esse fenômeno de dupla detecção, ele acontece pois o modelo atribui uma alta probabilidade para mais de uma classe em uma imagem e o algoritmo de supressão (Non-Maximum Suppression - NMS, utilizado pelo YOLO) não consegue definir claramente qual deve descartar.

---

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).
