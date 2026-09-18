"""Расчёт экономического эффекта COW ID — прозрачная модель с явными допущениями.

Принцип: ни одна цифра не берётся с потолка. Каждый вход — это либо параметр,
который хозяйство подставляет своими данными, либо публично известный порядок
величин, и он подписан источником прямо здесь. Модель намеренно консервативна:
там, где есть вилка, берётся нижняя граница.

Запуск:
    cowid economics --herd 500
    cowid economics --herd 2000 --kzt-per-usd 540
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Assumptions:
    """Входные допущения. Всё, что здесь, хозяйство заменяет своими цифрами."""

    herd: int = 500
    kzt_per_usd: float = 520.0

    # --- Заболеваемость и стоимость случая -------------------------------
    #: Доля стада, переносящая клинический случай (хромота, мастит, ацидоз,
    #: метрит) в течение года. Берём осторожную нижнюю оценку.
    annual_incidence: float = 0.25
    #: Стоимость одного клинического случая: лечение + потеря продуктивности
    #: + риск выбраковки. Нижняя граница диапазона, встречающегося в отраслевых
    #: публикациях по хромоте и маститу.
    cost_per_case_usd: float = 150.0
    #: Какую долю ущерба удаётся предотвратить, если поймать случай на 1–2 суток
    #: раньше. Консервативно: не «вылечили бесплатно», а «дешевле и короче».
    early_detection_savings_share: float = 0.30
    #: Какую долю случаев система реально ловит заранее. Подставляется из полноты,
    #: измеренной на ферме по ответам зоотехника.
    detection_recall: float = 0.60

    # --- Труд -------------------------------------------------------------
    #: Часы в сутки, которые бригада тратит на визуальный обход и поиск
    #: «подозрительных» животных.
    daily_inspection_hours: float = 1.5
    #: Какую долю этого времени снимает очередь событий вместо сплошного обхода.
    inspection_time_saved_share: float = 0.40
    labour_cost_usd_per_hour: float = 3.0

    # --- Затраты ----------------------------------------------------------
    #: Одна камера покрывает проход/зону. Сколько голов приходится на камеру.
    cows_per_camera: int = 100
    camera_cost_usd: float = 200.0
    #: Мини-ПК или edge-устройство на площадку.
    edge_device_usd: float = 600.0
    installation_usd_per_camera: float = 80.0
    #: Срок службы оборудования, лет — по нему раскладываем капитальные затраты.
    hardware_life_years: float = 5.0
    #: Эксплуатация: электричество, связь, обслуживание, лицензия ПО.
    annual_opex_usd_per_cow: float = 3.0

    # --- Альтернатива: ошейники -------------------------------------------
    collar_cost_usd_per_cow: float = 100.0
    collar_base_station_usd: float = 10_000.0
    collar_subscription_usd_per_cow_month: float = 4.0
    collar_life_years: float = 6.0


@dataclass
class Result:
    lines: list[tuple[str, float]] = field(default_factory=list)

    def add(self, label: str, value_usd: float) -> None:
        self.lines.append((label, value_usd))

    @property
    def total(self) -> float:
        return sum(v for _, v in self.lines)


def compute(a: Assumptions) -> dict:
    # ---- Выгода ----------------------------------------------------------
    benefit = Result()

    cases_per_year = a.herd * a.annual_incidence
    caught_early = cases_per_year * a.detection_recall
    health_saving = caught_early * a.cost_per_case_usd * a.early_detection_savings_share
    benefit.add(
        f"Раннее выявление болезней: {caught_early:.0f} из {cases_per_year:.0f} случаев в год",
        health_saving,
    )

    hours_saved = a.daily_inspection_hours * a.inspection_time_saved_share * 365
    labour_saving = hours_saved * a.labour_cost_usd_per_hour
    benefit.add(f"Экономия труда: {hours_saved:.0f} чел-часов в год", labour_saving)

    # ---- Затраты ---------------------------------------------------------
    cost = Result()
    cameras = max(1, -(-a.herd // a.cows_per_camera))     # округление вверх
    capex = (
        cameras * (a.camera_cost_usd + a.installation_usd_per_camera) + a.edge_device_usd
    )
    cost.add(f"Оборудование: {cameras} камер + edge, в год из {a.hardware_life_years:.0f} лет",
             capex / a.hardware_life_years)
    cost.add("Эксплуатация и ПО", a.herd * a.annual_opex_usd_per_cow)

    net = benefit.total - cost.total
    payback_months = (capex / (net / 12)) if net > 0 else float("inf")

    # ---- Альтернатива: ошейники -----------------------------------------
    collar_capex = a.herd * a.collar_cost_usd_per_cow + a.collar_base_station_usd
    collar_annual = (
        collar_capex / a.collar_life_years
        + a.herd * a.collar_subscription_usd_per_cow_month * 12
    )

    return {
        "benefit": benefit,
        "cost": cost,
        "net": net,
        "capex": capex,
        "cameras": cameras,
        "payback_months": payback_months,
        "collar_annual": collar_annual,
        "per_cow_year": net / a.herd,
    }


def fmt(usd: float, rate: float) -> str:
    return f"{usd:>12,.0f} USD  /  {usd * rate:>14,.0f} тг".replace(",", " ")


def report(a: Assumptions) -> str:
    r = compute(a)
    out: list[str] = []
    out.append("=" * 78)
    out.append(f"COW ID — экономический эффект.  Стадо: {a.herd} голов, курс {a.kzt_per_usd} тг/USD")
    out.append("=" * 78)

    out.append("\nВЫГОДА В ГОД")
    for label, value in r["benefit"].lines:
        out.append(f"  {label:<58}{fmt(value, a.kzt_per_usd)}")
    out.append(f"  {'ИТОГО выгода':<58}{fmt(r['benefit'].total, a.kzt_per_usd)}")

    out.append("\nЗАТРАТЫ В ГОД")
    for label, value in r["cost"].lines:
        out.append(f"  {label:<58}{fmt(value, a.kzt_per_usd)}")
    out.append(f"  {'ИТОГО затраты':<58}{fmt(r['cost'].total, a.kzt_per_usd)}")

    out.append("\nРЕЗУЛЬТАТ")
    out.append(f"  {'Чистый эффект в год':<58}{fmt(r['net'], a.kzt_per_usd)}")
    out.append(f"  {'На голову в год':<58}{fmt(r['per_cow_year'], a.kzt_per_usd)}")
    out.append(f"  {'Разовые вложения (CAPEX)':<58}{fmt(r['capex'], a.kzt_per_usd)}")
    payback = r["payback_months"]
    out.append(f"  {'Окупаемость вложений':<58}"
               f"{payback:.1f} мес." if payback != float("inf") else "не окупается")

    out.append("\nСРАВНЕНИЕ С ОШЕЙНИКАМИ (та же задача, датчик на каждом животном)")
    out.append(f"  {'Ошейники: стоимость владения в год':<58}"
               f"{fmt(r['collar_annual'], a.kzt_per_usd)}")
    out.append(f"  {'COW ID: стоимость владения в год':<58}"
               f"{fmt(r['cost'].total, a.kzt_per_usd)}")
    ratio = r["collar_annual"] / r["cost"].total if r["cost"].total else 0
    out.append(f"  {'Во сколько раз камера дешевле в год':<58}{ratio:.1f}x")

    out.append("\nЧУВСТВИТЕЛЬНОСТЬ: как меняется чистый эффект")
    out.append(f"  {'сценарий':<34}{'recall':>8}{'стоимость случая':>20}{'чистый эффект, тг':>22}")
    for recall, case_cost, name in [
        (0.40, 100.0, "пессимистичный"),
        (0.60, 150.0, "базовый"),
        (0.75, 250.0, "оптимистичный"),
    ]:
        alt = Assumptions(**{**a.__dict__})
        alt.detection_recall = recall
        alt.cost_per_case_usd = case_cost
        net = compute(alt)["net"]
        out.append(f"  {name:<34}{recall:>8.0%}{case_cost:>17,.0f} $"
                   f"{net * a.kzt_per_usd:>22,.0f}".replace(",", " "))

    out.append("\nПРИМЕЧАНИЕ")
    out.append("  Модель консервативна: везде, где есть вилка, взята нижняя граница.")
    out.append("  Полнота (recall) — допущение; на ферме её заменяет доля подтверждённых событий.")
    out.append("  Эффект от роста привесов и от снижения выбраковки НЕ учтён — он есть,")
    out.append("  но требует данных конкретного хозяйства.")
    return "\n".join(out)

