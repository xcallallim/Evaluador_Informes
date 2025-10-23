import pytest

from data.preprocessing.cleaner import Cleaner


@pytest.fixture()
def cleaner() -> Cleaner:
    return Cleaner()


@pytest.mark.parametrize(
    "original,expected",
    [
        ("Resumen ejecutivo - 2", "Resumen ejecutivo"),
        ("Conclusiones – 12", "Conclusiones"),
        ("Informe final 3", "Informe final"),
        ("Anexo Página 4 de 10", "Anexo"),
    ],
)
def test_strip_trailing_page_num_removes_page_indicators(cleaner: Cleaner, original: str, expected: str) -> None:
    assert cleaner._strip_trailing_page_num(original) == expected


def test_strip_trailing_page_num_keeps_non_page_suffix(cleaner: Cleaner) -> None:
    text = "Capítulo 5: Resultados finales"
    assert cleaner._strip_trailing_page_num(text) == text