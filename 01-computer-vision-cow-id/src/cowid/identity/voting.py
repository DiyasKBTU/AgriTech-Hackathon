"""Сколько даёт голосование по дорожке против решения по одному снимку.

В конвейере животное опознаётся не по кадру, а по всей дорожке: каждый кадр
голосует за ближайшую корову, если уверен, и решение принимается большинством.
Здесь это проверяется на реальных данных Cows2021.

Приближение к дорожке — снимки одной коровы за один день съёмки. Логика
голосования та же, что в `identity/identifier.py`.

Запуск (после `cowid train-reid` и `cowid eval-reid`):
    cowid voting-effect
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np


def measure(model_dir: Path, vote_ratio: float = 0.5) -> dict:
    import torch

    from .reid_dataset import Sample
    from .train import TrainConfig, _build_model, _extract

    ck = torch.load(model_dir / "encoder.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ck["config"])
    cfg.num_workers = 0
    split = json.loads((model_dir / "split.json").read_text(encoding="utf-8"))
    evaluation = json.loads((model_dir / "evaluation.json").read_text(encoding="utf-8"))
    threshold = 1.0 - evaluation["unknown_distance_at_far1"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, _ = _build_model(cfg, n_classes=1)
    encoder.load_state_dict(ck["state_dict"])
    encoder = encoder.to(device).eval()

    def samples(key: str) -> list[Sample]:
        return [Sample(Path(r["path"]), r["identity"]) for r in split[key]]

    query = samples("query")
    q_emb, q_ids = _extract(encoder, query, cfg, device)
    g_emb, g_ids = _extract(encoder, samples("gallery"), cfg, device)

    sim = q_emb @ g_emb.T
    g_ids = np.asarray(g_ids)
    best = sim.argmax(axis=1)
    best_id = g_ids[best]
    best_sim = sim[np.arange(len(q_ids)), best]
    confident = best_sim >= threshold

    single_ok = float(np.mean((best_id == np.asarray(q_ids)) & confident))
    single_wrong = float(np.mean((best_id != np.asarray(q_ids)) & confident))

    groups: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i, s in enumerate(query):
        groups[(s.identity, s.date)].append(i)

    correct = wrong = unknown = 0
    for (cow, _), idx in groups.items():
        votes = collections.Counter(best_id[i] for i in idx if confident[i])
        if not votes:
            unknown += 1
            continue
        top, n = votes.most_common(1)[0]
        if n / len(idx) < vote_ratio:
            unknown += 1
        elif top == cow:
            correct += 1
        else:
            wrong += 1

    total = len(groups)
    sizes = [len(g) for g in groups.values()]
    report = {
        "tracks": total,
        "images_per_track_median": int(np.median(sizes)),
        "similarity_threshold": threshold,
        "single": {"recognised": single_ok, "wrong": single_wrong,
                   "unknown": 1 - single_ok - single_wrong},
        "voting": {"recognised": correct / total, "wrong": wrong / total,
                   "unknown": unknown / total},
    }
    (model_dir / "voting_effect.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
