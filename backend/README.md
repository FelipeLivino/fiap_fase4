# CardioIA Backend

Backend Flask do prototipo mobile do IR ALEM 2.

## Como rodar

```bash
cd backend
python -m pip install -r requirements.txt
python app.py
```

Para ativar debug durante desenvolvimento:

```bash
set FLASK_DEBUG=1
python app.py
```

Endpoints:

- `GET /health`: verifica backend, labels e checkpoint.
- `GET /metrics`: retorna metricas do modelo final e metricas por patologia.
- `POST /predict`: recebe multipart/form-data com o campo `image` e retorna probabilidades por patologia.

O backend usa o checkpoint gerado pelo notebook:

```text
notebooks/artifacts_cardioia_pytorch_multilabel/modelos/vision_transformer_vit.pt
```

Os arquivos `.pt` nao devem ser versionados no GitHub comum porque ultrapassam o limite de 100 MB. Para reproduzir o backend em outra maquina, execute o notebook para gerar novamente os checkpoints ou compartilhe os pesos por Git LFS/Drive.

Este prototipo tem finalidade academica e nao deve ser usado como diagnostico medico.

## Docker

Pela raiz do projeto:

```bash
docker compose up --build
```

O compose expoe o backend em `http://localhost:5000` e monta os artefatos do notebook como volume somente leitura.
Abra o Docker Desktop antes de executar o comando.
