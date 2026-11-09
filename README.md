# Projeto NP2 — Redes Neurais com PyTorch

Classificação de imagens de vestuário utilizando o **FashionMNIST** e uma rede **Multi-Layer Perceptron (MLP)** implementada em PyTorch. Trabalho individual da disciplina Redes Neurais (2026.2).

## Objetivo
Classificar imagens de 28 × 28 pixels em 10 categorias de roupas e acessórios utilizando treinamento e validação manuais, regularização e avaliação em dados não vistos.

## Dados
FashionMNIST: 70.000 imagens, com 48.000 para treinamento, 12.000 para validação e 10.000 para teste. O dataset é baixado automaticamente pelo `torchvision`.

## Arquitetura
- Camadas lineares: **784 → 256 → 128 → 64 → 10**.
- **ReLU**, **BatchNorm1d** e **Dropout (30%)**.
- **243.530 parâmetros treináveis**.
- Classes personalizadas de `Dataset` e `nn.Module`, com `DataLoader` para mini-batches.

## Treinamento
- Loss: `CrossEntropyLoss`.
- Otimizador: `Adam`; learning rate `0.001`; weight decay `0.0001`.
- Batch size: `32`; até `50` épocas; Early Stopping com paciência de `5`.
- Loop manual com `zero_grad()`, forward, `backward()`, `step()`, `model.eval()` e `torch.no_grad()`.

## Resultados da execução documentada
| Métrica | Valor |
|---|---:|
| Épocas executadas | 24 |
| Melhor época | 19 |
| Melhor loss de validação | 0,2854 |
| Acurácia de validação na melhor época | 89,67% |
| Loss de teste | 0,3159 |
| Acurácia de teste | **88,45%** |
| F1-score macro de teste | **0,8839** |

Os resultados podem variar em novas execuções devido a processos aleatórios. O notebook contém os gráficos de loss/acurácia, matriz de confusão e análises.

## Como executar no Google Colab
1. Abra https://colab.research.google.com/.
2. Carregue `NP2_Redes_Neurais_FashionMNIST_GabrielMarcelino.ipynb`.
3. Execute todas as células em ordem, do começo ao fim.

## Conformidade com o enunciado
O Scikit-Learn foi utilizado apenas na divisão inicial dos dados e no cálculo de métricas. Não há estimadores preditivos do Scikit-Learn nem uso de Lightning, FastAI ou Keras para automatizar o treinamento.

## Arquivos
- `NP2_Redes_Neurais_FashionMNIST_GabrielMarcelino.ipynb` — notebook completo com saídas.
- `requirements.txt` — bibliotecas.
- `.gitignore` — exclusões para ambiente local.
