"""Реестр фермы: карточка коровы на дату из реальных записей MmCows.

Если датасет не распакован, тест пропускается — он проверяет чтение
настоящих записей, а не придуманных.
"""

from datetime import date

import pytest

from cowid import registry


pytestmark = pytest.mark.skipif(not (registry.HEALTH / "C07.csv").exists(),
                                reason="нет записей фермы MmCows")


def test_card_is_built_as_of_date_without_future_events():
    card = registry.cow_record("C07", date(2023, 7, 25))
    assert card.lactation and card.days_in_milk and card.eid
    assert card.pregnancy.startswith("стельная")
    leg = card.last_leg_problem
    assert leg is not None and leg.what == "хромота" and leg.day <= date(2023, 7, 25)
    assert all(e.day <= date(2023, 7, 25) for e in card.health + card.reproduction)


def test_lame_codes_are_translated_and_unknown_cow_is_none():
    assert "блок" in registry._lame_detail("Lame - BLKULRLJ")
    assert registry.cow_record("C99", date(2023, 7, 25)) is None
