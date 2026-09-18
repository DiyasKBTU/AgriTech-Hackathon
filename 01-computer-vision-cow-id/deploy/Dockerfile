# COW ID: образ для площадки (мини-ПК у камер) или центрального сервера.
#
# По умолчанию ставится torch для процессора. Для видеокарты:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 ...
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# libgl1 и libglib нужны OpenCV, который тянет ultralytics.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url "${TORCH_INDEX}" \
 && pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

# Данные, модели и база живут в томах, а не в образе: образ пересобирается,
# а накопленная история животных должна сохраниться.
VOLUME ["/app/var", "/app/models", "/app/configs", "/app/data"]

EXPOSE 8000
CMD ["cowid", "serve", "--host", "0.0.0.0", "--port", "8000", "--config", "configs/camera.yaml"]
