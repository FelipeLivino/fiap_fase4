# CardioIA Vision - PBL Fase 4

Projeto academico de Visao Computacional para apoio a triagem de achados em radiografias de torax usando o dataset NIH Chest X-rays.

## Entregas principais

- Notebook completo: `notebooks/CardioIA_Vision_PBL_Fase4.ipynb`
- Relatorio tecnico curto: `Docs/relatorio_tecnico_fase4.md`
- Prototipo mobile/backend: `Docs/ir_alem_2_mobile.md`
- Backend Flask: `backend/`
- App React Native/Expo: `mobile-app/`
- Docker Compose: `docker-compose.yml`

## Problema modelado

Classificacao multi-label de quatro patologias:

- `Infiltration`
- `Effusion`
- `Atelectasis`
- `Pneumothorax`

O problema foi tratado como multi-label porque uma radiografia pode conter mais de uma patologia simultaneamente.

## Resultado final

O modelo final selecionado foi o `Vision Transformer ViT`, com:

- F1 macro: `0,5426`
- Recall macro: `0,6605`
- AUC-ROC macro: `0,7201`
- PR-AUC macro: `0,5323`
- Hamming loss: `0,3328`
- Tempo medio: `0,0100 s/imagem`

## Como rodar com Docker

Abra o Docker Desktop e execute:

```bash
docker compose up --build
```

Servicos:

- Backend Flask: `http://localhost:5000`
- App web Expo: `http://localhost:8081`

O backend precisa do checkpoint local gerado pelo notebook em:

```text
notebooks/artifacts_cardioia_pytorch_multilabel/modelos/vision_transformer_vit.pt
```

Os arquivos `.pt` nao sao versionados no GitHub comum porque ultrapassam o limite de 100 MB.

## Observacao

Este projeto tem finalidade academica e educacional. Ele nao deve ser usado como ferramenta diagnostica real.
