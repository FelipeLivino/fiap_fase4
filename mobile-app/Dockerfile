FROM node:20-bookworm-slim

ENV EXPO_NO_TELEMETRY=1 \
    __UNSAFE_EXPO_HOME_DIRECTORY=/app/.expo-home \
    CI=1

WORKDIR /app

COPY mobile-app/package*.json ./
RUN npm ci --no-audit --no-fund --progress=false

COPY mobile-app ./

EXPOSE 8081 19000 19001 19002

CMD ["npm", "run", "start:docker"]
