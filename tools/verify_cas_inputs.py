"""Audit the external-CAS inputs without running Singular or Wolfram/Mathematica.

This script independently reconstructs the polynomial conditions from

    T * DF(x,k) * B = 0

using SymPy and compares them with the ideals stored in the Singular scripts.
It also checks the Wolfram Language scripts for the expected model, T, B, and
rate-constant order, verifies the frozen result-file component counts, and
checks the explicit two-pathway Component [6] reduction.

It is intended as a lightweight reproducibility check for reviewers/users who do
not have Singular or Wolfram/Mathematica installed.

Run from the repository root:

    python tools/verify_cas_inputs.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Shared symbols, deliberately created without assumptions so parsed expressions
# and direct expressions have the same symbol objects.
# ---------------------------------------------------------------------------
x1, x2, x3, x4, x5, x6 = sp.symbols("x1 x2 x3 x4 x5 x6")
p, q, r = sp.symbols("p q r")
t1, t2, t3, t4, t5, t6, t7, t8, t9 = sp.symbols("t1 t2 t3 t4 t5 t6 t7 t8 t9")
k1, k2, k3, k4, k5, km1, km2, km3, km4, km5 = sp.symbols(
    "k1 k2 k3 k4 k5 km1 km2 km3 km4 km5"
)

SYMBOLS = {
    str(s): s
    for s in (
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        p,
        q,
        r,
        t1,
        t2,
        t3,
        t4,
        t5,
        t6,
        t7,
        t8,
        t9,
        k1,
        k2,
        k3,
        k4,
        k5,
        km1,
        km2,
        km3,
        km4,
        km5,
    )
}

X = (x1, x2, x3, x4, x5, x6)
RATE_ORDER = (k1, k2, k3, k4, k5, km1, km2, km3, km4, km5)


def _normal_form(expr: sp.Expr) -> str:
    """Canonical-enough string for exact set comparison up to a global sign."""
    e = sp.factor(sp.expand(expr))
    s = str(e)
    if s.startswith("-"):
        e = -e
        s = str(sp.factor(e))
    return s.replace(" ", "")


def _assert_zero(expr: sp.Expr | sp.MatrixBase, label: str) -> None:
    if isinstance(expr, sp.MatrixBase):
        entries = list(expr)
    else:
        entries = [expr]
    bad = [sp.factor(sp.simplify(e)) for e in entries if sp.simplify(e) != 0]
    if bad:
        raise AssertionError(f"{label} is not zero; first nonzero entry: {bad[0]}")


def _split_top_level_commas(body: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
        else:
            current.append(ch)
    item = "".join(current).strip()
    if item:
        items.append(item)
    return items


def _parse_expr(text: str) -> sp.Expr:
    return sp.sympify(text.strip().replace("^", "**"), locals=SYMBOLS)


def singular_ideal_generators(path: Path) -> list[sp.Expr]:
    text = path.read_text()
    match = re.search(r"ideal\s+g\s*=\s*(.*?);\s*(?://|list|LIB|\")", text, re.S)
    if not match:
        raise AssertionError(f"Could not find `ideal g = ...;` in {path}")
    return [_parse_expr(item) for item in _split_top_level_commas(match.group(1))]


def coefficient_conditions(F: sp.Matrix, T: sp.Matrix, B: sp.Matrix, xvars: Sequence[sp.Symbol]) -> list[sp.Expr]:
    J = F.jacobian(xvars)
    U = sp.expand(T * J * B)
    conditions: list[sp.Expr] = []
    seen: set[str] = set()
    for entry in U:
        poly = sp.Poly(sp.expand(entry), *xvars)
        for coeff in poly.coeffs():
            coeff = sp.factor(sp.expand(coeff))
            if coeff == 0:
                continue
            key = _normal_form(coeff)
            if key not in seen:
                seen.add(key)
                conditions.append(coeff)
    return conditions


def compare_generator_sets(label: str, generated: Iterable[sp.Expr], scripted: Iterable[sp.Expr]) -> None:
    gen = {_normal_form(e) for e in generated if e != 0}
    scr = {_normal_form(e) for e in scripted if e != 0}
    if gen != scr:
        missing = sorted(gen - scr)
        extra = sorted(scr - gen)
        raise AssertionError(
            f"{label}: generated and scripted ideals differ.\n"
            f"  missing from script ({len(missing)}): {missing[:3]}\n"
            f"  extra in script ({len(extra)}): {extra[:3]}"
        )
    print(f"PASS: {label}: {len(gen)} generated conditions match the Singular ideal exactly.")


# ---------------------------------------------------------------------------
# Wolfram Language parsing checks
# ---------------------------------------------------------------------------

def _extract_wl_assignment(text: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*(.*?);", text, re.S)
    if not match:
        raise AssertionError(f"Could not find `{name} = ...;` in Wolfram script")
    return match.group(1).strip()


def _wl_to_python_expr(text: str) -> str:
    return text.replace("^", "**").replace("{", "[").replace("}", "]")


def parse_wl_expr(text: str) -> sp.Expr:
    return _parse_expr(_wl_to_python_expr(text))


def parse_wl_matrix(text: str) -> sp.Matrix:
    obj = eval(_wl_to_python_expr(text), {"__builtins__": {}}, SYMBOLS)  # noqa: S307 - controlled local file syntax
    return sp.Matrix(obj)


def parse_wl_list(text: str) -> tuple[sp.Symbol, ...]:
    obj = eval(_wl_to_python_expr(text), {"__builtins__": {}}, SYMBOLS)  # noqa: S307 - controlled local file syntax
    return tuple(obj)


def check_wolfram_script(
    label: str,
    path: Path,
    expected_F: sp.Matrix,
    expected_T: sp.Matrix,
    expected_B: sp.Matrix,
    expected_kvars: Sequence[sp.Symbol],
) -> None:
    text = path.read_text()
    T_wl = parse_wl_matrix(_extract_wl_assignment(text, "T"))
    B_wl = parse_wl_matrix(_extract_wl_assignment(text, "B"))
    kvars_wl = parse_wl_list(_extract_wl_assignment(text, "kvars"))
    F_wl = sp.Matrix([parse_wl_expr(_extract_wl_assignment(text, f"x{i}d")) for i in range(1, 7)])

    _assert_zero(T_wl - expected_T, f"{label} Wolfram T mismatch")
    _assert_zero(B_wl - expected_B, f"{label} Wolfram B mismatch")
    _assert_zero(F_wl - expected_F, f"{label} Wolfram vector field mismatch")
    if tuple(kvars_wl) != tuple(expected_kvars):
        raise AssertionError(f"{label} Wolfram kvars order mismatch: {kvars_wl}")
    _assert_zero(T_wl * B_wl, f"{label} Wolfram T.B")
    print(f"PASS: {label}: Wolfram script T, B, F, and k-variable order match the independent definitions.")


# ---------------------------------------------------------------------------
# Direct model definitions
# ---------------------------------------------------------------------------

def replication_model() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    F = sp.Matrix(
        [
            -k1 * x1 * x3 + km1 * x4 - k4 * x1 * x5 + km4 * x6,
            -k2 * x2 * x4 + km2 * x6 - k3 * x2 * x3 + km3 * x5,
            -k1 * x1 * x3 + km1 * x4 - k3 * x2 * x3 + km3 * x5 + 2 * k5 * x6 - 2 * km5 * x3**2,
            k1 * x1 * x3 - km1 * x4 - k2 * x2 * x4 + km2 * x6,
            k3 * x2 * x3 - km3 * x5 - k4 * x1 * x5 + km4 * x6,
            k2 * x2 * x4 - km2 * x6 + k4 * x1 * x5 - km4 * x6 - k5 * x6 + km5 * x3**2,
        ]
    )
    T = sp.Matrix(
        [
            [0, 0, 1, p, q, r],
            [1, -1, 0, 1, -1, 0],
            [0, 1, 1, 1, 2, 2],
        ]
    )
    B = sp.Matrix(
        [
            [p - 2, q - 1, r - 2],
            [p - 1, q - 2, r - 2],
            [-p, -q, -r],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    return F, T, B


def two_pathway_model() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    F = sp.Matrix(
        [
            -k1 * x1 * x2 + km1 * x3 - k3 * x1 * x2 + km3 * x4,
            -k1 * x1 * x2 + km1 * x3 - k3 * x1 * x2 + km3 * x4 + k5 * x5 - km5 * x2 * x6,
            k1 * x1 * x2 - km1 * x3 - k2 * x3 + km2 * x5,
            k3 * x1 * x2 - km3 * x4 - k4 * x4 + km4 * x5,
            k2 * x3 + k4 * x4 - km2 * x5 - km4 * x5 - k5 * x5 + km5 * x2 * x6,
            k5 * x5 - km5 * x2 * x6,
        ]
    )
    T = sp.Matrix(
        [
            [1, t1, 0, 0, t2, t3],
            [0, t4, 1, 0, t5, t6],
            [0, t7, 0, 1, t8, t9],
        ]
    )
    B = sp.Matrix(
        [
            [t1, t2, t3],
            [-1, 0, 0],
            [t4, t5, t6],
            [t7, t8, t9],
            [0, -1, 0],
            [0, 0, -1],
        ]
    )
    return F, T, B


# ---------------------------------------------------------------------------
# Result-file checks
# ---------------------------------------------------------------------------

def count_components(path: Path) -> int:
    text = path.read_text()
    return len(re.findall(r"^\[\d+\]:", text, flags=re.M))


def check_result_files() -> None:
    expected_counts = {
        ROOT / "results" / "replication_minAssChar_22_components.txt": 22,
        ROOT / "results" / "two_pathway_facstd_first5_components.txt": 5,
        ROOT / "results" / "two_pathway_elimination_minAssChar_components.txt": 6,
    }
    for path, expected in expected_counts.items():
        actual = count_components(path)
        if actual != expected:
            raise AssertionError(f"{path}: expected {expected} components, found {actual}")
        print(f"PASS: {path.relative_to(ROOT)} lists {actual} components as expected.")

    elim = (ROOT / "results" / "two_pathway_elimination_minAssChar_components.txt").read_text().replace(" ", "")
    required = ["_[1]=k2-k4+km1-km3", "_[2]=k3*km2-k1*km4"]
    for token in required:
        if token not in elim:
            raise AssertionError(f"Component [6] token missing from elimination output: {token}")
    print("PASS: two-pathway elimination result contains Component [6] conditions.")


def check_component_6_reduction(F: sp.Matrix) -> None:
    T6 = sp.Matrix(
        [
            [1, -1, 0, 0, 0, 1],
            [0, k1 / (k1 + k3), 1, 0, k1 / (k1 + k3), 0],
            [0, k3 / (k1 + k3), 0, 1, k3 / (k1 + k3), 0],
        ]
    )
    subs = {km3: km1 + k2 - k4, km2: k1 * km4 / k3}
    TF = sp.simplify(T6 * F.subs(subs))
    y2_expr = T6.row(1).dot(sp.Matrix(X))
    y3_expr = T6.row(2).dot(sp.Matrix(X))
    expected = sp.Matrix(
        [
            0,
            (km1 + k2) * (k1 * y3_expr - k3 * y2_expr) / (k1 + k3),
            -(km1 + k2) * (k1 * y3_expr - k3 * y2_expr) / (k1 + k3),
        ]
    )
    _assert_zero(sp.simplify(TF - expected), "two-pathway Component [6] reduced system")
    print("PASS: two-pathway Component [6] reduced system identity holds symbolically.")


def main() -> int:
    print("CAS input audit for the lumping reaction-network repo")
    print("=" * 72)

    F_rep, T_rep, B_rep = replication_model()
    _assert_zero(T_rep * B_rep, "self-replication T.B")
    _assert_zero(T_rep.row(1).dot(F_rep), "self-replication first integral row 2")
    _assert_zero(T_rep.row(2).dot(F_rep), "self-replication first integral row 3")
    print("PASS: self-replication T.B and prescribed first integrals are correct.")
    check_wolfram_script(
        "self-replication",
        ROOT / "mathematica" / "replication_critical_variety.wl",
        F_rep,
        T_rep,
        B_rep,
        RATE_ORDER,
    )
    rep_conditions = coefficient_conditions(F_rep, T_rep, B_rep, X)
    compare_generator_sets(
        "self-replication Section 6.1",
        rep_conditions,
        singular_ideal_generators(ROOT / "singular" / "replication_decomposition.sing"),
    )

    F_two, T_two, B_two = two_pathway_model()
    _assert_zero(T_two * B_two, "two-pathway T.B")
    _assert_zero(F_two[1] + F_two[2] + F_two[3] + F_two[4], "two-pathway total enzyme first integral")
    _assert_zero(F_two[0] + F_two[2] + F_two[3] + F_two[4] + F_two[5], "two-pathway total substrate/product first integral")
    print("PASS: two-pathway T.B and stoichiometric first integrals are correct.")
    check_wolfram_script(
        "two-pathway enzyme",
        ROOT / "mathematica" / "two_pathway_enzyme_critical_variety.wl",
        F_two,
        T_two,
        B_two,
        RATE_ORDER,
    )
    two_conditions = coefficient_conditions(F_two, T_two, B_two, X)
    compare_generator_sets(
        "two-pathway Section 6.2",
        two_conditions,
        singular_ideal_generators(ROOT / "singular" / "two_pathway_enzyme_decomposition.sing"),
    )
    check_component_6_reduction(F_two)
    check_result_files()

    print("=" * 72)
    print("All CAS input/output sanity checks passed.")
    print("Note: this script verifies the mathematical inputs and quoted result-file structure; it does not independently prove the full Singular prime decompositions are complete/minimal.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - helpful command-line audit output
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
