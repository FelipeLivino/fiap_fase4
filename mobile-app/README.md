# CardioIA Mobile

Prototipo React Native do IR ALEM 2 da FIAP. O app permite escolher uma imagem, enviar ao backend Flask e visualizar as categorias detectadas pela CNN/ViT treinada no notebook.

## Como rodar

1. Inicie o backend:

```bash
cd backend
python app.py
```

2. Em outro terminal, instale e abra o app:

```bash
cd mobile-app
npm install
npm run start:offline
```

Se o npm reclamar de permissao no cache do Windows, use cache local:

```bash
npm install --no-audit --no-fund --progress=false --cache .\.npm-cache
```

O script `start:offline` evita que o Expo dependa de chamadas externas durante a apresentacao local fora do Docker.

3. Configure a URL do backend no campo exibido no app.

- Em navegador web ou simulador iOS local: `http://localhost:5000`
- Em emulador Android: `http://10.0.2.2:5000`
- Em celular fisico: use o IP da maquina na rede, por exemplo `http://192.168.0.10:5000`

## Fluxo demonstrado no video

1. Abrir o app CardioIA.
2. Confirmar que o backend esta online.
3. Escolher uma imagem de raio-X.
4. Tocar em `Classificar`.
5. Mostrar as probabilidades por patologia e o resumo do modelo final.

Este app e um prototipo academico e nao deve ser usado para diagnostico medico.

## Docker

Pela raiz do projeto:

```bash
docker compose up --build mobile
```

O Expo fica na porta `8081`. O backend Flask fica em `http://localhost:5000`.
Abra o Docker Desktop antes de executar o comando.

No Docker, o Expo sobe em modo web para abrir a interface diretamente no navegador em `http://localhost:8081`.
