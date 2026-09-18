"""Командная строка COW ID. Все действия проекта — через одну команду `cowid`.

    ДАННЫЕ
    cowid download cows2021           скачать датасет (8 потоков, с докачкой)
    cowid verify cows2021             проверить архив и распакованные файлы
    cowid prepare cows2021            распаковать и показать, что внутри

    МОДЕЛИ
    cowid train-detector              детектор коров сверху (Cows2021, разметка рамок)
    cowid eval-detector               сравнить его с готовой моделью COCO
    cowid train-reid --data <папка>   модель узнавания (папка = корова)
    cowid eval-reid                   проверка на коровах, которых модель не видела
    cowid voting-effect               снимок против голосования по дорожке

    ФЕРМА
    cowid init-camera                 шаблон настроек камеры
    cowid enroll --data <папка>       зарегистрировать коров по фото (папка = корова)
    cowid process <видео> --day ...   обработать запись за сутки
    cowid check-video                 прогон на реальных роликах Cows2021
    cowid import-mmcows               реальные сутки 10 коров MmCows → база и проверка
    cowid mmcows-detector             детектор на другой ферме: до и после дообучения
    cowid serve                       платформа: сегодня, стадо, камера, качество
    cowid serve --phone               то же + адрес и https для камеры телефона
    cowid demo-video                  ролики для защиты (reports/demo/)
    cowid farm-chain                  вся цепочка на ферме MmCows: узнавание, IDF1, активность

    РАСЧЁТЫ
    cowid scenarios                   что замечает детектор отклонений
    cowid economics --herd 500        экономический эффект
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import PipelineConfig, ZoneConfig

app = typer.Typer(add_completion=False, help="COW ID — идентификация КРС и контроль активности")
console = Console()

MODELS_DIR = Path("models")
VAR_DIR = Path("var")


def _log(msg: str) -> None:
    console.print(msg, highlight=False)


def _dataset_name(dataset: str) -> str:
    from .datasets import DATASETS

    if dataset not in DATASETS:
        console.print(f"[red]Неизвестный датасет {dataset}.[/red] Есть: {', '.join(DATASETS)}")
        raise typer.Exit(1)
    return dataset


# --------------------------------------------------------------------------
# Данные
# --------------------------------------------------------------------------

@app.command()
def download(
    dataset: str = typer.Argument(..., help="cows2021, frontal_face, mmcows_sensor, mmcows_visual, mmcows_labels, mmcows_video_0725"),
    workers: int = typer.Option(8, help="Параллельных соединений (выше 8 прироста почти нет)"),
    repair: bool = typer.Option(False, help="Найти повреждённые участки и перекачать только их"),
):
    """Скачивает датасет. Остановить — Ctrl+C, продолжить — та же команда."""
    from . import datasets

    name = _dataset_name(dataset)
    code = (datasets.repair(name, workers, _log) if repair
            else datasets.download(name, workers, _log))
    raise typer.Exit(code)


@app.command()
def verify(
    dataset: str = typer.Argument(..., help="cows2021, frontal_face, mmcows_sensor, mmcows_visual, mmcows_labels, mmcows_video_0725"),
    archive_only: bool = typer.Option(False, help="Не проверять распакованные файлы"),
):
    """Проверяет, что данные целы: архив против контрольных сумм издателя,
    затем каждый распакованный файл против архива."""
    from . import datasets

    name = _dataset_name(dataset)
    if datasets.verify_archive(name, _log) != 0:
        raise typer.Exit(1)
    root = datasets.TARGET_DIR / name
    if archive_only or not root.exists():
        return
    _log(f"Сверяю распакованные файлы в {root} с архивом...")
    report = datasets.verify_extracted(datasets.archive_path(name), root, _log)
    bad = len(report["missing"]) + len(report["damaged"])
    style = "red" if bad else "green"
    _log(f"[{style}]Файлов {report['files']}, нет на диске {len(report['missing'])}, "
         f"не совпала сумма {len(report['damaged'])}[/{style}] ({report['seconds']} с)")
    for p in (report["missing"] + report["damaged"])[:10]:
        _log(f"  {p}")
    if bad:
        _log(f"Распаковать заново: cowid prepare {name} --force")
        raise typer.Exit(1)


@app.command()
def prepare(
    dataset: str = typer.Argument(..., help="cows2021, frontal_face, mmcows_sensor, mmcows_visual, mmcows_labels, mmcows_video_0725"),
    force: bool = typer.Option(False, help="Распаковать заново, даже если уже распаковано"),
    shrink: int = typer.Option(0, help="Сделать уменьшенную копию фото (длинная сторона, px) "
                                       "в data/prepared/<датасет>_<px>"),
):
    """Распаковывает скачанный архив и показывает, что внутри."""
    from . import datasets

    name = _dataset_name(dataset)
    archive = datasets.archive_path(name)
    target = datasets.TARGET_DIR / name
    if not archive.exists():
        console.print(f"[red]Нет архива {archive}.[/red] Скачайте: cowid download {name}")
        raise typer.Exit(1)
    if not datasets.is_complete(name):
        console.print("[yellow]Архив скачан не полностью.[/yellow] "
                      f"Продолжить: cowid download {name}")
        raise typer.Exit(1)

    if target.exists() and any(target.iterdir()) and not force:
        console.print(f"Уже распаковано: {target}")
    else:
        target.mkdir(parents=True, exist_ok=True)
        with console.status(f"Распаковка {archive.name}..."):
            with zipfile.ZipFile(archive) as z:
                z.extractall(target)
        console.print(f"[green]Распаковано:[/green] {target}")

    _describe_dataset(target)
    if shrink:
        small = Path("data/prepared") / f"{name}_{shrink}"
        with console.status(f"Уменьшенная копия в {small}..."):
            n = datasets.shrink_images(target, small, shrink, _log)
        _log(f"[green]Уменьшенная копия:[/green] {small}, снимков {n}")


def _describe_dataset(root: Path) -> None:
    """Ищет папки, где лежат снимки по животным, и печатает сводку."""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    per_folder: Counter = Counter()
    videos = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in suffixes:
            per_folder[p.parent] += 1
        elif p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            videos += 1

    # «Папка с животными» — та, у которой много подпапок со снимками.
    id_parents: Counter = Counter(folder.parent for folder in per_folder)

    table = Table(title="Где лежат снимки по животным")
    table.add_column("Папка")
    table.add_column("Подпапок", justify="right")
    table.add_column("Снимков", justify="right")
    for folder, n_ids in id_parents.most_common(6):
        images = sum(c for f, c in per_folder.items() if f.parent == folder)
        table.add_row(str(folder), str(n_ids), str(images))
    console.print(table)
    console.print(f"Видеофайлов: {videos}")


# --------------------------------------------------------------------------
# Модели
# --------------------------------------------------------------------------

@app.command("train-detector")
def train_detector_cmd(
    epochs: int = typer.Option(40),
    base_model: str = typer.Option("yolo26n-obb.pt", help="Предобученная OBB-модель"),
    imgsz: int = typer.Option(640),
    batch: int = typer.Option(32),
    workers: int = typer.Option(4),
    device: str = typer.Option("auto"),
    out: Path = typer.Option(MODELS_DIR / "detector"),
):
    """Обучает детектор коров «вид сверху» на размеченных кадрах Cows2021
    и проверяет его на кадрах 4–11 марта, которых он не видел."""
    from .detect.detectors import resolve_device
    from .detect.train_detector import prepare_dataset, train_detector

    dev = resolve_device(device)
    counts = prepare_dataset(log=_log)
    weights = train_detector(Path(counts["data_yaml"]), out, base_model, epochs, imgsz,
                             batch, workers, dev, log=_log)
    # Функцию-команду вызываем со всеми аргументами: значения по умолчанию
    # у typer — служебные объекты, а не числа.
    eval_detector(weights=weights, conf=0.35, baseline="yolo26n.pt", device=device)


@app.command("eval-detector")
def eval_detector(
    weights: Path = typer.Option(MODELS_DIR / "detector" / "cow_obb.pt"),
    conf: float = typer.Option(0.35, help="Порог уверенности, как в конвейере"),
    baseline: str = typer.Option("yolo26n.pt", help="Готовая модель COCO для сравнения"),
    device: str = typer.Option("auto"),
):
    """Сравнивает обученный детектор с готовой моделью COCO на одних и тех же
    проверочных кадрах."""
    from .detect.detectors import resolve_device
    from .detect.train_detector import (
        PREPARED, evaluate_coco_baseline, evaluate_detector, save_report,
    )

    dev = resolve_device(device)
    with console.status("Проверка обученного детектора..."):
        ours = evaluate_detector(weights, PREPARED / "data.yaml", conf, dev)
    with console.status(f"Проверка готовой модели {baseline}..."):
        coco = evaluate_coco_baseline(baseline, conf, dev)
    report = {"trained": ours, "coco_baseline": coco}
    save_report(report, weights.parent / "evaluation.json")

    table = Table(title=f"Детекция коров сверху: {ours['oriented']['images']} кадров "
                        f"4–11 марта, коров {ours['oriented']['cows']}")
    for col in ("Модель", "Найдено", "Пропущено", "Лишних рамок", "Полнота", "Точность"):
        table.add_column(col, justify="left" if col == "Модель" else "right")
    rows = [
        ("обученная, повёрнутые рамки", ours["oriented"]),
        ("обученная, прямоугольники", ours["axis_aligned"]),
        ("готовая COCO, прямоугольники", coco["iou50"]),
        ("готовая COCO, центр внутри рамки", coco["center_inside"]),
    ]
    for name, s in rows:
        table.add_row(name, str(s["found"]), str(s["missed"]), str(s["extra"]),
                      f"{s['recall']:.1%}", f"{s['precision']:.1%}")
    console.print(table)
    _log(f"mAP50 {ours['mAP50']:.3f}, mAP50-95 {ours['mAP50_95']:.3f} "
         f"(совпадение — IoU ≥ 0.5, порог уверенности {conf})")
    _log(f"Отчёт: {weights.parent / 'evaluation.json'}")


@app.command("train-reid")
def train_reid(
    data: Path = typer.Option(..., help="Папка, где одна подпапка = одно животное"),
    out: Path = typer.Option(MODELS_DIR / "reid", help="Куда сохранить модель"),
    backbone: str = typer.Option("resnet50"),
    epochs: int = typer.Option(20),
    batch_size: int = typer.Option(48),
    image_size: int = typer.Option(224),
    lr: float = typer.Option(3e-4),
    test_identity_ratio: float = typer.Option(0.3, help="Доля ЖИВОТНЫХ для проверки"),
    max_images_per_identity: int = typer.Option(60),
    workers: int = typer.Option(4),
    device: str = typer.Option("auto"),
):
    """Обучает модель узнавания животных.

    Проверка идёт на животных, которых модель не видела ни разу: датасет делится
    по особям, а не по снимкам. Иначе в проверку попадут снимки тех же коров,
    модель их просто запомнит, и высокая точность ничего не будет значить.
    """
    from .identity.train import TrainConfig
    from .identity.train import train as run_training

    cfg = TrainConfig(
        data_root=str(data), out_dir=str(out), backbone=backbone, epochs=epochs,
        batch_size=batch_size, image_size=image_size, lr=lr,
        test_identity_ratio=test_identity_ratio,
        max_images_per_identity=max_images_per_identity,
        num_workers=workers, device=device,
    )
    console.print(Panel.fit(
        f"[bold]Обучение узнавания[/bold]\n"
        f"данные: {data}\nмодель: {backbone}, эпох: {epochs}\n"
        f"проверка на {test_identity_ratio:.0%} животных, не участвующих в обучении",
        border_style="cyan",
    ))
    report = run_training(cfg, log=_log)
    _print_reid(report["best"])
    _log(f"[green]Модель:[/green] {out / 'encoder.pt'}")


@app.command("eval-reid")
def eval_reid(
    model: Path = typer.Option(MODELS_DIR / "reid", help="Папка с обученной моделью"),
    controls: bool = typer.Option(False, help="Контроль: необученная сеть и закрытый центр снимка"),
):
    """Проверяет модель узнавания на животных, которых она не видела,
    и подбирает порог «не знаю» (путаница с чужими не чаще 1 из 100)."""
    from .identity.train import evaluate_saved_model

    if controls:
        from .identity.train import control_checks

        r = control_checks(model)
        table = Table(title="Контрольная проверка: не завышена ли цифра")
        for col in ("Вариант", "Rank-1", "mAP"):
            table.add_column(col, justify="left" if col == "Вариант" else "right")
        names = {"trained_mask0": "обученная модель",
                 "untrained_mask0": "необученная сеть (ImageNet)",
                 "trained_mask60": "обученная, центр снимка закрыт",
                 "untrained_mask60": "необученная, центр закрыт"}
        for key, label in names.items():
            table.add_row(label, f"{r[key]['rank1']:.1%}", f"{r[key]['mAP']:.3f}")
        console.print(table)
        _log("Если необученная сеть почти так же хороша — проверка слишком лёгкая.\n"
             "Если закрытый центр почти ничего не меняет — модель узнаёт по фону.")
        return

    report = evaluate_saved_model(model)
    _print_reid(report)
    _log("\nРекомендуемый порог «не знаю» для конфигурации камеры:\n"
         f"  gallery.unknown_distance: {report['unknown_distance_at_far1']:.3f}")
    (model / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _print_reid(m: dict) -> None:
    table = Table(title="Узнавание животных, которых модель не видела")
    table.add_column("Показатель")
    table.add_column("Значение", justify="right")
    table.add_column("Что значит")
    table.add_row("Rank-1", f"{m.get('rank1', 0):.1%}",
                  "ближайший похожий снимок — то же животное")
    table.add_row("Rank-5", f"{m.get('rank5', 0):.1%}",
                  "правильное животное среди пяти ближайших")
    table.add_row("mAP", f"{m.get('mAP', 0):.3f}", "качество всего списка похожих")
    table.add_row("TAR при 1% ложных", f"{m.get('tar@far0.01', 0):.1%}",
                  "сколько узнаётся, если путать чужих не чаще 1 из 100")
    table.add_row("Животных в проверке", str(m.get("n_test_identities", 0)), "")
    console.print(table)


@app.command("voting-effect")
def voting_effect(
    model: Path = typer.Option(MODELS_DIR / "reid"),
    vote_ratio: float = typer.Option(0.5),
):
    """Сравнивает решение по одному снимку с голосованием по дорожке."""
    from .identity.voting import measure

    r = measure(model, vote_ratio)
    table = Table(title=f"Дорожек (корова + день): {r['tracks']}, "
                        f"снимков в дорожке: медиана {r['images_per_track_median']}")
    for col in ("", "Узнано", "Чужой номер", "«Не знаю»"):
        table.add_column(col, justify="left" if not col else "right")
    for label, key in (("по одному снимку", "single"), ("голосованием", "voting")):
        s = r[key]
        table.add_row(label, f"{s['recognised']:.1%}", f"{s['wrong']:.2%}",
                      f"{s['unknown']:.1%}")
    console.print(table)


# --------------------------------------------------------------------------
# Ферма
# --------------------------------------------------------------------------

@app.command("init-camera")
def init_camera(
    path: Path = typer.Argument(Path("configs/camera.yaml")),
    camera_id: str = typer.Option("camera-1"),
):
    """Создаёт шаблон настроек камеры: зоны, калибровка, пороги."""
    cfg = PipelineConfig(camera_id=camera_id)
    cfg.zones = [
        ZoneConfig(name="feeder", polygon=[(0, 0), (640, 0), (640, 120), (0, 120)]),
        ZoneConfig(name="drinker", polygon=[(560, 300), (640, 300), (640, 420), (560, 420)]),
    ]
    cfg.save(path)
    _log(f"[green]Шаблон:[/green] {path}")
    _log("Отредактируйте полигоны зон под кадр вашей камеры "
         "(координаты в пикселях) и масштаб calibration.pixels_per_meter.")


@app.command()
def enroll(
    data: Path = typer.Option(..., help="Папка, где одна подпапка = одна корова"),
    config: Path = typer.Option(Path("configs/camera.yaml")),
    per_cow: int = typer.Option(20, help="Сколько снимков брать на корову"),
    prefix: str = typer.Option("", help="Приставка к номеру, например KZ-"),
):
    """Регистрирует коров по готовым снимкам: имя подпапки становится номером.

    Нужна там, где бирки не читаются камерой: на старте, если на ферме уже
    есть фото стада, или для проверки на открытом датасете.
    """
    import cv2

    from .identity.embedder import build_embedder
    from .identity.gallery import BiometricGallery
    from .identity.reid_dataset import index_by_folder
    from .types import BBox

    cfg = PipelineConfig.load(config) if config.exists() else PipelineConfig()
    embedder = build_embedder(cfg.embedder)
    gallery_path = VAR_DIR / f"gallery_{cfg.camera_id}.json"
    gallery = (BiometricGallery.load(gallery_path, cfg.gallery) if gallery_path.exists()
               else BiometricGallery(cfg.gallery))

    by_cow: dict[str, list] = {}
    for s in index_by_folder(data):
        by_cow.setdefault(s.identity, []).append(s)
    with console.status(f"Регистрация {len(by_cow)} коров..."):
        for cow, samples in sorted(by_cow.items()):
            samples = sorted(samples, key=lambda s: (s.date, str(s.path)))
            step = max(1, len(samples) // per_cow)
            vectors = []
            for s in samples[::step][:per_cow]:
                img = cv2.imread(str(s.path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                vec = embedder.embed(img, BBox(0, 0, w - 1, h - 1))
                if vec is not None:
                    vectors.append(vec)
            gallery.enroll(f"{prefix}{cow}", vectors, source="photo")
    VAR_DIR.mkdir(parents=True, exist_ok=True)
    gallery.save(gallery_path)
    _log(f"[green]В галерее {gallery.size()} коров[/green], "
         f"портретов {gallery.total_embeddings()}: {gallery_path}")


@app.command()
def process(
    video: Path = typer.Argument(..., help="Видеофайл за сутки"),
    day: str = typer.Option(date.today().isoformat(), help="Дата записи, ГГГГ-ММ-ДД"),
    config: Path = typer.Option(Path("configs/camera.yaml")),
    max_frames: Optional[int] = typer.Option(None, help="Ограничить число кадров"),
    evidence: bool = typer.Option(True, help="Сохранить кадры-доказательства"),
):
    """Обрабатывает запись камеры за сутки и сохраняет результат в базу."""
    from .evidence import EvidenceCollector
    from .pipeline import CowIdPipeline
    from .store.db import Store

    if not config.exists():
        _log(f"[red]Нет настроек камеры {config}.[/red] Создайте их: cowid init-camera {config}")
        raise typer.Exit(1)
    cfg = PipelineConfig.load(config)
    store = Store(cfg.storage.database)
    pipeline = CowIdPipeline(cfg, store=store,
                             gallery_path=VAR_DIR / f"gallery_{cfg.camera_id}.json")
    the_day = date.fromisoformat(day)

    with console.status(f"Обработка {video.name}..."):
        result = pipeline.process_video(video, the_day, max_frames=max_frames)

    if evidence and result.events:
        collector = EvidenceCollector(cfg, cfg.storage.evidence_dir)
        found = collector.collect(result.tracklets, the_day, video,
                                  cow_ids={e.cow_id for e in result.events})
        for cow_id, ev in found.items():
            store.save_evidence(cow_id, the_day, ev.as_dict())

    ident = result.identifier_stats
    _log(
        f"Кадров обработано: {result.frames_processed} "
        f"({result.processing_fps:.1f} в секунду)\n"
        f"Треков: {len(result.tracklets)}, опознано: {result.identified} "
        f"(по бирке {ident.get('by_tag', 0)}, по внешности {ident.get('by_biometric', 0)}, "
        f"«не знаю» {ident.get('unknown', 0)})\n"
        f"Животных в галерее: {result.gallery_size}\n"
        f"Событий: {len(result.events)}"
    )
    for e in result.events:
        style = {"alert": "red", "warning": "yellow"}.get(e.severity, "cyan")
        _log(f"  [{style}]{e.severity}[/{style}] {e.cow_id}: {e.title}")


COWS2021 = Path("data/real/cows2021/4vnrca7qw1642qlwxjadp87h7/Sub-levels/Identification")


@app.command("check-video")
def check_video(
    videos: Path = typer.Option(COWS2021 / "Videos", help="Папка с роликами"),
    photos: Path = typer.Option(COWS2021 / "Test", help="Фото, по которым регистрировали коров"),
    config: Path = typer.Option(Path("configs/cows2021.yaml")),
    limit: Optional[int] = typer.Option(None, help="Сколько роликов взять"),
    min_track: int = typer.Option(8, help="Короче этого дорожки не считаются"),
    sheets: int = typer.Option(40, help="Сколько дорожек положить на листы сверки"),
    out: Path = typer.Option(Path("reports/video_check")),
):
    """Прогон конвейера на реальных роликах без ответов: сколько дорожек узнано,
    сколько раз одна корова оказалась в двух местах сразу, листы для сверки глазами.

    Перед этим: cowid enroll --data <photos> --config <config>
    """
    from . import video_check

    cfg = PipelineConfig.load(config)
    gallery = VAR_DIR / f"gallery_{cfg.camera_id}.json"
    if not gallery.exists():
        _log(f"[red]Нет галереи {gallery}.[/red] Сначала: "
             f"cowid enroll --data \"{photos}\" --config {config}")
        raise typer.Exit(1)
    clips = sorted(videos.rglob("*.avi")) + sorted(videos.rglob("*.mp4"))
    if limit:
        clips = clips[:limit]
    _log(f"Роликов: {len(clips)}")
    r = video_check.run(cfg, clips, gallery, out, min_track=min_track, log=_log)

    split_file = Path(cfg.embedder.weights or "").parent / "split.json"
    if split_file.exists():
        split = json.loads(split_file.read_text(encoding="utf-8"))
        seen = {s["identity"] for s in split["train"]}
        tracks = json.loads((out / "tracks.json").read_text(encoding="utf-8"))
        named = [t for t in tracks if t["cow_id"]]
        r["identified_seen_in_training"] = sum(t["cow_id"] in seen for t in named)
        r["identified_unseen_in_training"] = sum(t["cow_id"] not in seen for t in named)
        (out / "summary.json").write_text(json.dumps(r, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    pages = video_check.contact_sheets(out, photos, n_tracks=sheets)
    table = Table(title=f"Реальные ролики: {r['clips']}, кадров обработано {r['frames_processed']}")
    table.add_column("Показатель")
    table.add_column("Значение", justify="right")
    table.add_row(f"Дорожек (не короче {min_track} кадров)", str(r["tracks"]))
    table.add_row("Узнано", f"{r['identified']} ({r['identified'] / max(1, r['tracks']):.0%})")
    table.add_row("«Не знаю»", str(r["unknown"]))
    table.add_row("Разных коров названо", str(r["cows_named"]))
    table.add_row("Одна корова в двух местах сразу (пар)", str(r["conflicts"]))
    table.add_row("Скорость, кадров в секунду", str(r["fps"]))
    if "identified_seen_in_training" in r:
        table.add_row("  из узнанных: коровы, на которых училась модель",
                      str(r["identified_seen_in_training"]))
        table.add_row("  из узнанных: коровы, которых модель не видела",
                      str(r["identified_unseen_in_training"]))
    console.print(table)
    _log(f"Листы сверки: {', '.join(str(p) for p in pages)}")


@app.command()
def serve(
    config: Path = typer.Option(Path("configs/mmcows_sensors.yaml"),
                                help="по умолчанию — демо на реальных сутках MmCows"),
    host: str = typer.Option("127.0.0.1", help="0.0.0.0 — открыть другим устройствам в сети"),
    port: int = typer.Option(8000),
    phone: bool = typer.Option(False, "--phone",
                               help="подключить телефон: адрес в сети и https для его камеры"),
    https_port: int = typer.Option(8443, help="порт https для телефона"),
):
    """Платформа: сегодня, стадо, корова, живая камера, качество моделей."""
    import threading

    import uvicorn

    from .api.app import create_app, lan_addresses, phone_certificate

    if not config.exists():
        _log(f"[red]Нет файла настроек {config}[/red]")
        raise typer.Exit(1)
    cfg = PipelineConfig.load(config)
    application = create_app(cfg)
    _log(f"[green]Платформа:[/green] http://127.0.0.1:{port}  "
         f"(настройки {config}, база {cfg.storage.database})")
    if phone:
        host = "0.0.0.0"
        addresses = lan_addresses()
        cert = phone_certificate(Path("var/tls"), addresses)
        application.state.phone = {"http_port": port, "https_port": https_port if cert else None}
        if cert:
            server = uvicorn.Server(uvicorn.Config(
                application, host=host, port=https_port, log_level="warning",
                ssl_certfile=str(cert[0]), ssl_keyfile=str(cert[1])))
            threading.Thread(target=server.run, daemon=True).start()
            for ip in addresses:
                _log(f"[green]С телефона:[/green] https://{ip}:{https_port}/#/live")
            _log("Телефон — в той же сети Wi-Fi (или ноутбук подключён к точке доступа телефона). "
                 "Браузер предупредит о сертификате: «Дополнительно» → «Перейти на сайт». "
                 "Windows может спросить про брандмауэр — разрешить для частной сети.")
        else:
            _log("[yellow]Не нашёл openssl — https не поднят, камера телефона в браузере "
                 "не включится. Можно подключить телефон приложением IP Webcam по ссылке.[/yellow]")
            for ip in addresses:
                _log(f"Адрес в сети: http://{ip}:{port}")
    try:
        uvicorn.run(application, host=host, port=port, log_level="warning")
    finally:
        from .live import ENGINE

        ENGINE.stop()


@app.command("demo-video")
def demo_video(
    only: Optional[str] = typer.Option(None, help="known, unseen или other_farm"),
):
    """Демо-ролики для защиты: кадр с рамками и номерами + панель показателей.
    Результат — reports/demo/*.mp4 (H.264, открывается в любом проигрывателе)."""
    from . import demo_video as dv

    if only:
        dv.render(only, log=_log)
    else:
        dv.render_all(log=_log)


@app.command("import-mmcows")
def import_mmcows(
    config: Path = typer.Option(Path("configs/mmcows_sensors.yaml")),
    fresh: bool = typer.Option(True, help="Начать базу заново (ответы зоотехника в ней пропадут)"),
):
    """Реальные сутки 10 коров MmCows (датчики) → база платформы и проверка
    детектора отклонений: разброс, число тревог, совпадение с журналом фермы."""
    from .mmcows import import_and_check, save_report
    from .store.db import Store

    cfg = PipelineConfig.load(config) if config.exists() else PipelineConfig()
    db = Path(cfg.storage.database)
    if fresh and db.exists():
        db.unlink()
    store = Store(db)
    report = import_and_check(store, cfg.baseline, log=_log)
    save_report(report, Path("reports/mmcows/detector_on_sensors.json"))
    _log(report["summary_text"])
    table = Table(title="События детектора на реальных данных")
    for col in ("Корова", "Сутки", "Уровень", "Признак", "Журнал фермы в этот день"):
        table.add_column(col)
    for e in report["events"]:
        table.add_row(e["cow"], e["day"], e["severity"], e["title"], e["farm_note_same_day"])
    console.print(table)
    _log(f"База: {db}. Платформа: cowid serve --config {config}")


@app.command("mmcows-detector")
def mmcows_detector_cmd(
    epochs: int = typer.Option(25),
    imgsz: int = typer.Option(1280, help="1600 не помещается в 8 ГБ видеопамяти"),
    batch: int = typer.Option(8),
    device: str = typer.Option("auto"),
    skip_train: bool = typer.Option(False, help="Взять уже обученные веса"),
):
    """Детектор коров на другой ферме (MmCows): до дообучения, дообучение
    на утренних кадрах, проверка на вечерних; «стоит/лежит» по корове."""
    from .detect import mmcows_detector as md
    from .detect.detectors import resolve_device

    dev = resolve_device(device)
    if not (md.PREPARED / "data.yaml").exists():
        md.prepare(log=_log)
    before = [
        md.evaluate("models/detector/cow_obb.pt", None, imgsz, 0.35, dev,
                    name="детектор «вид сверху» (Cows2021)", is_obb=True, log=_log),
        md.evaluate("models/pretrained/yolo26n.pt", [19], imgsz, 0.3, dev,
                    name="готовая модель COCO", log=_log),
    ]
    weights = Path("models/detector_mmcows/cow_lying.pt")
    if not skip_train or not weights.exists():
        weights = md.train("yolo26n.pt", epochs, imgsz, batch, dev, log=_log)
    after = md.evaluate(str(weights), None, imgsz, 0.35, dev,
                        name="дообученная на этой ферме", trained=True, log=_log)
    report = {"before": before, "after": after}
    report["summary_text"] = md.summary_text(report)
    out = Path("reports/mmcows/lying_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(report["summary_text"])


@app.command("farm-chain")
def farm_chain_cmd(
    step: str = typer.Option("all", help="all | crops | train | frames"),
    epochs: int = typer.Option(8),
    force: bool = typer.Option(False, help="Пересчитать всё заново"),
):
    """Вся цепочка ТЗ на ферме MmCows с ответами: узнавание 16 коров,
    tracking (IDF1, подмены номера), часы лёжа конкретной коровы против людей."""
    from . import mmcows_chain as fc

    if step == "crops":
        fc.prepare_crops(log=_log)
    elif step == "train":
        fc.train_reid(epochs=epochs, log=_log)
    elif step == "frames":
        fc.enroll_gallery(fc.unknown_threshold(), log=_log)
        fc.run_frames(log=_log)
    else:
        fc.run_all(epochs=epochs, force=force, log=_log)


# --------------------------------------------------------------------------
# Расчёты
# --------------------------------------------------------------------------

@app.command()
def scenarios(
    cv: list[float] = typer.Option([0.08, 0.12, 0.18],
                                   help="Суточный разброс показателей здоровой коровы"),
    herd: int = typer.Option(200),
):
    """Что детектор отклонений замечает, а что нет, при заданном разбросе.

    Это проверка логики детектора на описанных зоотехнических сценариях,
    а не на данных фермы. Что она доказывает и чего нет — в docs/ПРОЕКТ.md.
    """
    from .anomaly.scenarios import SCENARIOS, run_healthy, run_scenario
    from .config import BaselineConfig

    cfg = BaselineConfig()
    table = Table(title=f"Детектор отклонений, стадо {herd} голов, 14 спокойных суток до эпизода")
    table.add_column("Сценарий")
    for c in cv:
        table.add_column(f"разброс {c:.0%}", justify="right")
    for sc in SCENARIOS:
        cells = []
        for c in cv:
            r = run_scenario(sc, cfg, daily_cv=c, herd=herd)
            if sc.expected_status == "alert":
                day = f", {r.median_detection_day:.0f}-е сут" if r.median_detection_day else ""
                cells.append(f"поймано {r.correct_rate:.0%}{day}")
            else:
                cells.append(f"верно {r.correct_rate:.0%}")
        table.add_row(sc.name, *cells)
    table.add_row("Ложные тревоги", *[
        f"{run_healthy(cfg, daily_cv=c, herd=herd).per_100_cows_month:.1f} на 100 гол/мес"
        for c in cv
    ])
    console.print(table)
    _log("«поймано 95%, 2-е сут» — тревога поднялась у 95% животных с таким эпизодом,\n"
         "в среднем на вторые сутки. Для охоты и одного плохого дня верная реакция —\n"
         "не поднимать тревогу.")


@app.command()
def economics(
    herd: int = typer.Option(500, help="Размер стада, голов"),
    kzt_per_usd: float = typer.Option(520.0),
    recall: Optional[float] = typer.Option(None, help="Доля болезней, пойманных заранее"),
):
    """Экономический эффект на стадо: выгода, затраты, сравнение с ошейниками."""
    from .economics import Assumptions, report

    a = Assumptions(herd=herd, kzt_per_usd=kzt_per_usd)
    if recall is not None:
        a.detection_recall = recall
    print(report(a))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
