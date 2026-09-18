"""Датасеты: загрузка с докачкой и проверка целостности.

Зачем многопоточность. Серверы ограничивают скорость на одно соединение
(замерено: ~0.8 МБ/с), а не ваш канал. Восемь соединений качают разные куски
файла параллельно — скорость вырастает в 4–5 раз.

Как сохраняется прогресс. Файл делится на куски по 32 МБ. Готовый кусок
сначала принудительно сбрасывается на диск и только потом отмечается в журнале
`<файл>.progress.json`. Порядок важен: отметка «готово» никогда не опережает
реальную запись, даже при внезапном отключении питания. Остановить можно в
любой момент; повторный запуск докачает только недостающее.

Как проверяется, что ничего не потеряно.
  1. Если издатель опубликовал контрольную сумму (Zenodo — MD5), весь файл
     сверяется с ней. Совпала — файл байт в байт тот, что на сервере.
  2. У каждого файла внутри zip своя контрольная сумма CRC32; архив читается
     целиком, и каждая сумма сверяется.
  3. После распаковки `verify_extracted` сверяет CRC32 каждого файла на диске
     с архивом — так ловится порча при распаковке.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class Remote:
    filename: str
    url: str
    #: Контрольные суммы от издателя, если он их публикует.
    md5: Optional[str] = None
    sha256: Optional[str] = None


HF = "https://huggingface.co/datasets/neis-lab/mmcows/resolve/main/"

DATASETS = {
    # Бристольский университет. Коровы сверху: разметка рамок, 181 корова
    # для узнавания, 301 видео. Лицензия некоммерческая.
    "cows2021": Remote(
        "cows2021.zip",
        "https://data.bris.ac.uk/datasets/tar/4vnrca7qw1642qlwxjadp87h7.zip",
    ),
    # Zenodo 10535934. Морды 459 коров крупным планом. Лицензия CC-BY 4.0.
    "frontal_face": Remote(
        "cows_frontal_face.zip",
        "https://zenodo.org/records/10535934/files/INDIVIDUAL%20SUBJECTS%20Data.zip?download=1",
        md5="1327f35502f85b69807a56060591e5c9",
    ),
    # MmCows (Purdue, NeurIPS 2024), лицензия CC BY-NC-SA 4.0. Весь датасет —
    # терабайты; берём только нужное. Hugging Face публикует SHA-256 файлов.
    # 14 суток датчиков по 10 коровам: лежание, положение, температура, удой.
    "mmcows_sensor": Remote(
        "mmcows_sensor_data.zip", HF + "sensor_data.zip",
        sha256="66b8d251f59834055820bc5242c2f7ddc31ca1de63ca9423550a25959fa2e608",
    ),
    # Один день, 20 000 кадров с 4 камер: рамки, номера коров, «лежит/стоит».
    "mmcows_visual": Remote(
        "mmcows_visual_data.zip", HF + "visual_data.zip",
        sha256="b0afbf109aab1c32c548d675c2f5f655a23a86aef18b4a3ddb39892b34442350",
    ),
    # Поведение по секундам (ходьба, стояние, кормление, питьё, лежание).
    "mmcows_labels": Remote(
        "mmcows_behavior_labels.zip", HF + "behavior_labels.zip",
        sha256="587a93b3705579ee5ff121173fd61f023b10b4c8ed1ad0fed72f1d5380773246",
    ),
    # Видео за 25.07 (кадр раз в 15 с, 4 камеры в одном кадре) — другая ферма
    # и другой ракурс, для демонстрации на незнакомом видео.
    "mmcows_video_0725": Remote(
        "mmcows_0725.mp4", HF + "15s_interval_combined_videos/0725.mp4",
        sha256="f01dd1b2fb4939037814c3fef841011650f4c756ba1201eb43668498ec471b10",
    ),
}

CHUNK = 32 * 1024 * 1024
TARGET_DIR = Path("data/real")

Log = Callable[[str], None]


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


class Journal:
    """Журнал готовых кусков. Пишется атомарно: сначала во временный файл,
    потом подменой — так обрыв посреди записи не испортит журнал."""

    def __init__(self, path: Path, total_chunks: int):
        self.path = path
        self.total = total_chunks
        self.done: set[int] = set()
        self._lock = threading.Lock()
        if path.exists():
            self.done = set(json.loads(path.read_text()).get("done", []))

    def mark(self, index: int) -> None:
        with self._lock:
            self.done.add(index)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(json.dumps({"total": self.total, "done": sorted(self.done)}))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)


def archive_path(name: str) -> Path:
    return TARGET_DIR / DATASETS[name].filename


def journal_path(target: Path) -> Path:
    return target.with_name(target.name + ".progress.json")


def is_complete(name: str) -> bool:
    journal = journal_path(archive_path(name))
    if not journal.exists():
        return archive_path(name).exists()
    state = json.loads(journal.read_text())
    return len(state.get("done", [])) >= state.get("total", 1)


def download(name: str, workers: int = 8, log: Log = print) -> int:
    remote = DATASETS[name]
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    target = TARGET_DIR / remote.filename
    jpath = journal_path(target)

    size = remote_size(remote.url)
    total_chunks = (size + CHUNK - 1) // CHUNK
    journal = Journal(jpath, total_chunks)

    # Файл уже частично скачан другим способом, а журнала нет —
    # засчитываем всё, что целиком лежит в начале файла.
    if target.exists() and not jpath.exists():
        have = target.stat().st_size
        for i in range(have // CHUNK):
            journal.done.add(i)
        if journal.done:
            journal.mark(max(journal.done))
            log(f"Найден недокачанный файл: {have / 1e9:.2f} ГБ засчитано как готовое")

    # Файл сразу доводим до полного размера: каждый поток пишет в свой участок.
    # На Windows это занимает минуту-другую: система заполняет место нулями.
    if not target.exists() or target.stat().st_size != size:
        log(f"Резервирую место под файл: {size / 1e9:.2f} ГБ...")
        with open(target, "ab") as f:
            f.truncate(size)

    todo = [i for i in range(total_chunks) if i not in journal.done]
    log(f"{remote.filename}: {size / 1e9:.2f} ГБ, кусков {total_chunks}, "
        f"осталось {len(todo)}, потоков {workers}")
    if not todo:
        return verify_archive(name, log)

    file_lock = threading.Lock()
    started = time.time()
    fetched = 0
    fetched_lock = threading.Lock()

    def fetch_chunk(index: int) -> int:
        nonlocal fetched
        start = index * CHUNK
        end = min(start + CHUNK, size) - 1
        for attempt in range(8):
            try:
                req = urllib.request.Request(remote.url,
                                             headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    if r.status != 206:
                        raise IOError(f"сервер вернул {r.status} вместо 206")
                    data = r.read()
                if len(data) != end - start + 1:
                    raise IOError(f"получено {len(data)} байт вместо {end - start + 1}")
                with file_lock:
                    with open(target, "r+b") as f:
                        f.seek(start)
                        f.write(data)
                        # Сброс на диск ДО отметки в журнале. Иначе при
                        # отключении питания журнал сказал бы «готово», а на
                        # диске остались бы нули — и архив молча оказался бы битым.
                        f.flush()
                        os.fsync(f.fileno())
                journal.mark(index)
                with fetched_lock:
                    fetched += len(data)
                return index
            except Exception as exc:  # noqa: BLE001 — сеть рвётся по-разному
                wait = min(60, 2 ** attempt)
                print(f"  кусок {index}: {exc}; повтор через {wait} с", file=sys.stderr)
                time.sleep(wait)
        raise RuntimeError(f"кусок {index} не скачался за 8 попыток")

    # Пул без `with` намеренно: выход из `with` ждёт ВСЕ задачи, и по Ctrl+C
    # загрузка продолжилась бы до конца. Здесь очередь отменяется сразу.
    pool = ThreadPoolExecutor(workers)
    futures = [pool.submit(fetch_chunk, i) for i in todo]
    try:
        for _ in as_completed(futures):
            elapsed = time.time() - started
            rate = fetched / elapsed / 1e6 if elapsed else 0
            done_n = len(journal.done)
            left_mb = (total_chunks - done_n) * CHUNK / 1e6
            eta = left_mb / rate / 60 if rate else 0
            print(f"\r  готово {done_n}/{total_chunks} "
                  f"({done_n / total_chunks:.0%})  {rate:.2f} МБ/с  "
                  f"осталось ~{eta:.0f} мин   ", end="", flush=True)
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        log(f"\nОстановлено. Прогресс сохранён: {len(journal.done)}/{total_chunks}. "
            f"Чтобы продолжить, запустите ту же команду.")
        return 1
    pool.shutdown(wait=True)

    failed = [f for f in futures if f.exception() is not None]
    if failed:
        log(f"\nНе скачалось кусков: {len(failed)}. Запустите команду ещё раз — "
            f"докачаются только они.")
        return 1

    print()
    return verify_archive(name, log)


# --------------------------------------------------------------------------
# Проверка
# --------------------------------------------------------------------------

def file_hash(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_archive(name: str, log: Log = print) -> int:
    """Архив скачан — проверяем, что он тот самый и открывается."""
    remote = DATASETS[name]
    target = archive_path(name)
    for algorithm, expected in (("md5", remote.md5), ("sha256", remote.sha256)):
        if not expected:
            continue
        log(f"Сверяю {algorithm.upper()} всего файла с опубликованным...")
        actual = file_hash(target, algorithm)
        if actual != expected:
            log(f"{algorithm.upper()} не совпал: {actual}, ожидалось {expected}")
            log(f"Скачать заново повреждённое: cowid download {name} --repair")
            return 1
        log(f"{algorithm.upper()} совпал: {actual}")
    if not target.name.endswith(".zip"):
        log(f"Готово, файл цел: {target}")
        return 0
    log("Проверяю контрольные суммы файлов внутри архива...")
    try:
        with zipfile.ZipFile(target) as z:
            bad = z.testzip()
        if bad:
            log(f"Повреждён файл внутри архива: {bad}")
            log(f"Починить: cowid download {name} --repair")
            return 1
    except Exception as exc:  # noqa: BLE001 — zipfile и zlib бросают разное
        log(f"Архив повреждён: {exc}")
        log(f"Починить: cowid download {name} --repair")
        return 1
    log(f"Готово, архив цел: {target}")
    return 0


def verify_extracted(archive: Path, root: Path, log: Log = print) -> dict:
    """Сверяет каждый распакованный файл с архивом: есть ли он и совпадает ли CRC32."""
    with zipfile.ZipFile(archive) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
    missing: list[str] = []
    damaged: list[str] = []
    started = time.time()
    for n, info in enumerate(infos, 1):
        path = root / info.filename
        if not path.exists():
            missing.append(info.filename)
            continue
        crc = 0
        with open(path, "rb") as f:
            while chunk := f.read(8 * 1024 * 1024):
                crc = zlib.crc32(chunk, crc)
        if crc != info.CRC:
            damaged.append(info.filename)
        if n % 10000 == 0:
            log(f"  проверено {n}/{len(infos)}")
    return {
        "files": len(infos),
        "missing": missing,
        "damaged": damaged,
        "seconds": round(time.time() - started),
    }


def _damaged_chunks(target: Path, log: Log) -> set[int]:
    """Куски, в которых повреждено содержимое архива.

    Каждый файл внутри zip хранит контрольную сумму. Читаем все файлы подряд,
    и где сумма не сошлась, по положению файла в архиве вычисляем куски для
    перекачки. Так вместо всего архива перекачивается только испорченное.
    """
    damaged: set[int] = set()
    with zipfile.ZipFile(target) as z:
        infos = sorted(z.infolist(), key=lambda i: i.header_offset)
        size = target.stat().st_size
        total = len(infos)
        for n, info in enumerate(infos, start=1):
            try:
                with z.open(info) as f:
                    while f.read(8 * 1024 * 1024):
                        pass
                continue
            except Exception:  # noqa: BLE001 — любая ошибка чтения значит порчу
                pass
            start = info.header_offset
            end = infos[n].header_offset if n < total else size
            damaged.update(range(start // CHUNK, (end - 1) // CHUNK + 1))
            log(f"  повреждён: {info.filename}")
    return damaged


def repair(name: str, workers: int = 8, log: Log = print) -> int:
    target = archive_path(name)
    jpath = journal_path(target)
    log(f"Проверяю каждый файл внутри {target.name}...")
    try:
        damaged = _damaged_chunks(target, log)
    except zipfile.BadZipFile as exc:
        log(f"Оглавление архива повреждено ({exc}) — починка по частям невозможна.")
        return 1
    if not damaged:
        log("Повреждений не найдено.")
        return verify_archive(name, log)

    log(f"Повреждённых кусков: {len(damaged)} "
        f"({len(damaged) * CHUNK / 1e9:.2f} ГБ) — перекачиваю только их.")
    state = json.loads(jpath.read_text())
    state["done"] = [i for i in state["done"] if i not in damaged]
    jpath.write_text(json.dumps(state))
    return download(name, workers, log)


def shrink_images(src: Path, dst: Path, long_side: int, log: Log = print) -> int:
    """Уменьшенная копия снимков с той же структурой папок.

    Фото с телефона весят по 5 МБ (4000x3000). Сеть всё равно видит 224x224,
    а распаковка JPEG такого размера на каждой эпохе занимает больше времени,
    чем само обучение. Копия делается один раз.
    """
    from PIL import Image, ImageOps

    count = 0
    for path in sorted(src.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not path.is_file():
            continue
        target = dst / path.relative_to(src).with_suffix(".jpg")
        if target.exists():
            count += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((long_side, long_side))
            img.save(target, quality=92)
        count += 1
        if count % 500 == 0:
            log(f"  уменьшено {count}")
    return count
