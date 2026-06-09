# IR ALEM 2 - Prototipo Mobile CardioIA

Este documento resume o prototipo criado para o IR ALEM 2: um app em React Native integrado a um backend Flask que carrega o modelo final treinado no notebook.

## O que foi implementado

- App React Native com Expo em `mobile-app/`.
- Tela de selecao/upload de imagem.
- Chamada HTTP para backend Flask.
- Exibicao das quatro categorias avaliadas:
  - `Infiltration`
  - `Effusion`
  - `Atelectasis`
  - `Pneumothorax`
- Exibicao das probabilidades por classe e das principais metricas do modelo final.
- Backend Flask em `backend/`.
- Endpoint real de inferencia usando o checkpoint `vision_transformer_vit.pt` gerado pelo notebook.

## Como executar

Terminal 1:

```bash
cd backend
python -m pip install -r requirements.txt
python app.py
```

Terminal 2:

```bash
cd mobile-app
npm install
npm run start:offline
```

Se houver erro de permissao no cache do npm no Windows, instale com cache local:

```bash
npm install --no-audit --no-fund --progress=false --cache .\.npm-cache
```

O script `start:offline` evita falhas por chamadas externas do Expo durante a demonstracao local fora do Docker.

## Como executar com Docker

Na raiz do projeto:

```bash
docker compose up --build
```

Antes de executar, confirme que o Docker Desktop esta aberto e com o engine Linux ativo.

Servicos expostos:

- Backend Flask: `http://localhost:5000`
- Expo/Metro: `http://localhost:8081`

O container do backend monta `notebooks/artifacts_cardioia_pytorch_multilabel` como volume somente leitura. Isso evita copiar o checkpoint grande para dentro da imagem e mantem o modelo usado no app igual ao gerado pelo notebook.
No Docker, o Expo sobe em modo web para abrir a interface diretamente no navegador em `http://localhost:8081`. Para demonstracao mobile real, tambem e possivel rodar `npm run start:offline` fora do Docker e abrir pelo Expo Go.

Os checkpoints `.pt` ficam fora do GitHub comum porque alguns arquivos ultrapassam 100 MB. Eles devem ser gerados pelo notebook ou armazenados separadamente via Git LFS/Drive.

No app, ajuste a URL do backend:

- Web/local: `http://localhost:5000`
- Emulador Android: `http://10.0.2.2:5000`
- Celular fisico: `http://IP_DA_MAQUINA:5000`

## Roteiro sugerido para video de ate 3 minutos

1. Mostrar rapidamente o notebook e dizer que o modelo final selecionado foi o `Vision Transformer ViT`.
2. Abrir o terminal com o backend Flask rodando.
3. Abrir o app React Native.
4. Conferir o status online do backend.
5. Selecionar uma imagem de raio-X.
6. Tocar em `Classificar`.
7. Explicar que o app mostra probabilidades multi-label para as quatro patologias.
8. Finalizar destacando que e um prototipo academico e nao substitui diagnostico medico.

## Observacao de governanca

O app apresenta resultados de um modelo treinado em dataset publico e deve ser usado apenas para demonstracao academica. Qualquer aplicacao real exigiria validacao clinica externa, revisao especialista, monitoramento de vieses e governanca formal.
