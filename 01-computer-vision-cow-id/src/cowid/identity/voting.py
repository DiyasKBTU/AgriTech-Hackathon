"""Сколько даёт голосование по дорожке против решения по одному снимку.

В конвейере животное опознаётся не по кадру, а по всей дорожке: каждый кадр
голосует за ближайшую корову, если уверен, и решение принимается большинством.
Здесь это проверяется на реальных данных Cows2021.

Приближение к дорожке — снимки одной коровы за один день съёмки. Логика
голосования та же, что в `identity/identifier.py`.

Запуск (после `cowid train-reid` и `cowid eval-reid`):
    cowid voting-effect

Здесь же `assign_unique` — раздача номеров всем дорожкам кадра сразу, чтобы
одна корова не стояла в двух местах одной камеры.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Optional

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


# --------------------------------------------------------------------------
# Один номер — одна рамка
# --------------------------------------------------------------------------

def assign_unique(tallies: list[dict], totals: list[float], min_votes: float = 3,
                  min_share: float = 0.5) -> list[Optional[str]]:
    """Номера дорожек одной камеры в одном кадре: каждая корова — не больше одной рамки.

    Без этого правила каждая дорожка решает сама, и одна корова то и дело
    оказывается в двух местах одной камеры (на MmCows — в 44% кадров, половина
    ошибок номера). Здесь номера раздаются сразу всем дорожкам кадра: сумма
    долей голосов максимальна, корова достаётся одной дорожке, проигравшая
    получает свой следующий номер, если за него хватает голосов, иначе «?».

    `tallies[i]` — голоса дорожки по коровам (вес), `totals[i]` — все её голоса,
    включая «не знаю». Номер возможен, если за него не меньше `min_votes`
    голосов (или все голоса короткой дорожки) и не меньше `min_share` от всех.
    """
    from scipy.optimize import linear_sum_assignment

    n = len(tallies)
    cows = sorted({c for t in tallies for c in t if c is not None})
    if not n or not cows:
        return [None] * n
    # Столбцы: коровы, затем по «пустому» месту на каждую дорожку (номер не дан).
    cost = np.zeros((n, len(cows) + n))
    cost[:, :len(cows)] = 1e6
    for i, (tally, total) in enumerate(zip(tallies, totals)):
        for j, cow in enumerate(cows):
            votes = tally.get(cow, 0.0)
            if total > 0 and votes >= min(min_votes, total) and votes / total >= min_share:
                cost[i, j] = -votes / total
    rows, cols = linear_sum_assignment(cost)
    out: list[Optional[str]] = [None] * n
    for i, j in zip(rows, cols):
        if j < len(cows) and cost[i, j] < 0:
            out[i] = cows[j]
    return out
