"""Check-digit algorithms referenced *by name* from locale-pack data files.

A pack declares `"checksum": "luhn"`; adding a country that uses an existing algorithm is pure data.
Each algorithm: compute(body) -> check string, validate(full) -> bool.
"""

from __future__ import annotations

from typing import Callable

_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
      [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
      [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
      [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
      [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]
_DNI = "TRWAGMYFPDXBNJZSQVHLCKE"
_CN_W = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
_CN_C = "10X98765432"


def _digits(s: str) -> list[int]:
    return [int(c) for c in s if c.isdigit()]


def _luhn_sum(digs: list[int]) -> int:
    total = 0
    for i, d in enumerate(reversed(digs)):
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total


def luhn_compute(body: str) -> str:
    return str((10 - _luhn_sum(_digits(body) + [0]) % 10) % 10)


def verhoeff_compute(body: str) -> str:
    c = 0
    for i, d in enumerate(reversed(_digits(body))):
        c = _D[c][_P[(i + 1) % 8][d]]
    return str(_INV[c])


def verhoeff_validate(full: str) -> bool:
    c = 0
    for i, d in enumerate(reversed(_digits(full))):
        c = _D[c][_P[i % 8][d]]
    return c == 0


def dni_compute(body: str) -> str:
    return _DNI[int("".join(map(str, _digits(body)))) % 23]


def insee_compute(body: str) -> str:
    return f"{97 - int(''.join(map(str, _digits(body)))) % 97:02d}"


def iso7064_compute(body: str) -> str:
    d = _digits(body)
    return _CN_C[sum(w * x for w, x in zip(_CN_W, d)) % 11]


def _validate_via_compute(compute: Callable[[str], str], check_len: int) -> Callable[[str], bool]:
    def validate(full: str) -> bool:
        clean = "".join(c for c in full if c.isalnum())
        if len(clean) <= check_len:
            return False
        body, check = clean[:-check_len], clean[-check_len:]
        try:
            return compute(body) == check.upper()
        except (ValueError, IndexError):
            return False
    return validate


ALGORITHMS: dict[str, tuple[Callable[[str], str], Callable[[str], bool]]] = {
    "luhn": (luhn_compute, _validate_via_compute(luhn_compute, 1)),
    "verhoeff": (verhoeff_compute, verhoeff_validate),
    "dni": (dni_compute, _validate_via_compute(dni_compute, 1)),
    "insee": (insee_compute, _validate_via_compute(insee_compute, 2)),
    "iso7064_11_2": (iso7064_compute, _validate_via_compute(iso7064_compute, 1)),
}


def compute_check(name: str, body: str) -> str:
    return ALGORITHMS[name][0](body)


def validate_checksum(name: str, value: str) -> bool:
    return ALGORITHMS[name][1](value)
