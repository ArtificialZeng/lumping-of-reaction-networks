# Lumping of reaction networks (Python)

A Python package for analyzing **exact linear lumping** of parameter-dependent **mass action** reaction networks (and closely related polynomial ODE models).

This repository accompanies the submitted manuscript:

> **"Lumping of reaction networks: Generic and critical parameters"** *(submitted manuscript)*

The software implements the main computational ingredients from the paper:

1. **Generic (parameter-independent) lumpings** (paper Section 3):
   - **Type 1:** eliminate *common non-reactant species*;
   - **Type 2:** project along *stoichiometric first integrals*.
2. **Critical parameters** for a prescribed lumping map **T** (paper Section 4):
   algebraic conditions on the rate constants under which **T** becomes an exact linear lumping.
3. **Proper lumpings** (paper Section 5):
   partition-based lumping maps with a dedicated Jacobian block / column-sum criterion.


### Scope

The Python package implements the mass-action / polynomial ODE computations used in the examples: generic Type 1/Type 2 lumping, the Li--Rabitz critical-parameter condition `T*DF*B = 0`, proper-lumping column-sum tests, reduced-polynomial-system construction for small examples, and numerical validation.

The manuscript also discusses product-form kinetic extensions and invariant-theoretic symmetry ideas. Those parts are theoretical in this repository; they are not yet exposed as a separate generalized-kinetics API.

In addition to these core computations, the package provides small “quality of life” utilities:

- a reaction-network string parser (`"A + B <-> C"`),
- compact reporting helpers for candidate reductions,
- optional numerical validation for small symbolic problems.

---

## Contents

- [Installation](#installation)
- [If you are new to Python](#if-you-are-new-to-python)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Core workflows](#core-workflows)
  - [1) Define a network](#1-define-a-network)
  - [2) Compute generic lumpings](#2-compute-generic-lumpings)
  - [3) Critical parameters for a prescribed T](#3-critical-parameters-for-a-prescribed-t)
  - [4) Constrained lumping](#4-constrained-lumping)
  - [5) Proper lumping (partitions)](#5-proper-lumping-partitions)
  - [6) Enumerate candidate reductions and print a report](#6-enumerate-candidate-reductions-and-print-a-report)
- [Computer algebra workflows](#computer-algebra-workflows)
  - [Row-echelon ansatz enumeration (general linear lumpings)](#row-echelon-ansatz-enumeration-general-linear-lumpings)
  - [Component enumeration and decomposition](#component-enumeration-and-decomposition)
- [Mathematica / Wolfram Language workflow](#mathematica--wolfram-language-workflow)
- [Reproducibility artifacts](#reproducibility-artifacts)
- [Verification notes](#verification-notes)
- [Numerical validation](#numerical-validation)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [How to cite](#how-to-cite)

---

## Installation

### Requirements

- Python **≥ 3.8**
- SymPy **≥ 1.12**
- NumPy **≥ 1.20**
- SciPy **≥ 1.7**

### Install (recommended: editable install)

From the repository root:

```bash
pip install -e .
```

That installs the `lumping_analysis` package so you can import it from anywhere.

---

## If you are new to Python

If you do not typically work in Python, this is the minimum setup that tends to work well.

### 1) Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install the package

```bash
pip install -e .
```

### 3) Run an example script

```bash
python examples/example_4_2_three_species.py
python examples/example_constrained_michaelis_menten.py
```

If you see printed conditions and/or a reduced model description, your environment is working.

---

## Repository layout

- `src/lumping_analysis/`
  - `network.py`, `reaction.py`, `parser.py`: define reaction networks and mass-action ODEs
  - `analyzer.py`: generic lumping, critical-parameter computations, proper lumping
  - `report.py`: compact console/Markdown reporting utilities
- `examples/`: runnable scripts aligned with paper examples and common workflows
- `tests/`: a small test suite (useful as executable documentation)
- `mathematica/`: corresponding Mathematica / Wolfram Language workflow (see [section below](#mathematica--wolfram-language-workflow))
- `singular/`: Singular scripts decomposing the Section 6 critical-parameter ideals (`facstd`, `minAssChar`)
- `results/`: frozen expected CAS outputs quoted in the manuscript appendices
- `figures/`: source scripts and generated PDFs for the manuscript figures
- `notes/`: short SymPy scripts verifying selected mathematical claims in the manuscript (see [section below](#verification-notes))
- `tools/`: repository-level audit scripts, including a CAS-input checker that does not require Singular or Wolfram/Mathematica
- `basic_usage.py`: a minimal one-file usage demo

### Example scripts

The `examples/` directory contains runnable scripts that mirror common use-cases
and (where applicable) align with examples in the manuscript.

- `examples/example_4_2_three_species.py`: three-species linear network (paper Section 4.4, three-species example)
- `examples/example_constrained_michaelis_menten.py`: constrained reduction for reversible Michaelis–Menten (paper Section 4.4, label `ex:mm`)
- `examples/example_proper_lumping_michaelis_menten.py`: proper lumping criterion demonstration (paper Section 5)
- `examples/search_proper_lumpings_michaelis_menten.py`: enumerate proper lumpings (small search)
- `examples/example_section_6_gpl_replication.py`: self-replication case study setup (paper Section 6.1)
- `examples/example_section_6_two_pathway_enzyme.py`: two-pathway enzyme case study, including the explicit Component [6] reduction (paper Section 6.2)
- `examples/enumerate_reductions_michaelis_menten.py`: batch enumeration + report printing

You can run any of these from the repository root via `python examples/<script>.py`.

---

## Quick start

```python
import sympy as sp
from lumping_analysis import michaelis_menten_network, LumpingAnalyzer

# Built-in network (reversible Michaelis–Menten)
net = michaelis_menten_network()
print(net.summary())

an = LumpingAnalyzer(net)

# Generic lumpings (Type 1 and Type 2)
generic = an.find_generic_lumpings()
print("Common non-reactants:", generic["common_non_reactants"])
print("Stoichiometric first integrals:")
for mu in generic["stoichiometric_integrals"]:
    print("  ", mu)

# Critical parameters for a prescribed candidate lumping matrix T
# (variables ordered as [S, E, C, P] in the built-in model)
T = sp.Matrix([
    [1, 0, 1, 1],  # S + C + P
    [0, 1, 1, 0],  # E + C
    [0, 0, 1, 0],  # C
])

res = an.find_critical_parameters(T)
print("Critical conditions:")
for c in res["conditions"]:
    print("  ", sp.factor(c), "= 0")
```

---

## Core workflows

### 1) Define a network

#### A. From reaction strings (recommended)

```python
from lumping_analysis import ReactionNetwork

net = ReactionNetwork.from_string("A + B <-> C")

# Multiple reactions (semicolon or newline separated)
net = ReactionNetwork.from_string(
    """
    S + E <-> C
    C <-> E + P
    """,
    species_names=["S", "E", "C", "P"],
)
```

#### B. Programmatically (full control)

```python
import sympy as sp
from lumping_analysis import Reaction, ReactionNetwork

k1, km1 = sp.symbols("k1 km1", positive=True)

# Species ordering: [A, B, C]
reactions = [
    Reaction((1, 1, 0), (0, 0, 1), k1),   # A + B -> C
    Reaction((0, 0, 1), (1, 1, 0), km1),  # C -> A + B
]

net = ReactionNetwork(n_species=3, reactions=reactions, species_names=["A", "B", "C"])
```

---

### 2) Compute generic lumpings

For **generic parameters** (paper Section 3), exact linear lumpings are restricted to the “obvious” families:

- eliminate *common non-reactants* (Type 1), and/or
- use *stoichiometric first integrals* (Type 2).

```python
generic = an.find_generic_lumpings()

print("Common non-reactant species:", generic["common_non_reactants"])
print("Stoichiometric first integrals:")
for mu in generic["stoichiometric_integrals"]:
    print(mu)
```

The function returns both human-readable species names (when available) and indices.

---

### 3) Critical parameters for a prescribed T

The critical-parameter test implemented here is the Jacobian invariance condition:

- Let **T** be an `e × n` full-row-rank matrix and let columns of **B** be a basis of `ker(T)`.
- Then **T** is a lumping map for the specialization `x' = F(x,k*)` iff

```text
T · DF(x,k*) · B = 0    for all x.
```

Computationally:

1. build the symbolic Jacobian `J = DF(x,k)`,
2. compute a symbolic kernel basis `B` for `ker(T)`,
3. form `TJB = T*J*B`,
4. extract **polynomial coefficient conditions** in the state variables `x`.

```python
import sympy as sp

T = sp.Matrix([
    [1, 0, 1, 1],
    [0, 1, 1, 0],
])

res = an.find_critical_parameters(T)
print("Conditions:")
for c in res["conditions"]:
    print("  ", sp.factor(c), "= 0")

# If the conditions are linear in the rate constants, the code also returns
# an rref-based analysis of the induced linear system.
analysis = res["solutions"].get("linear_system_analysis")
if analysis:
    print("Linear relations among rate constants:")
    for r in analysis["relations"]:
        print("  ", r, "= 0")
```

### Lumping-adapted coordinates (coordinate completion)

Sometimes it is useful to explicitly rewrite the ODE in coordinates where the first `e` variables are the lumped variables `y = T x`.

Given a candidate lumping matrix `T` (shape `e×n`), the standard procedure is:

1. complete `T` to an invertible `n×n` matrix `T*` by appending suitable rows,
2. define the change of variables `y = T* x`,
3. transform the vector field:

```text
y' = H(y,k) = T* F(T*^{-1} y, k).
```

The package provides a direct helper:

```python
out = an.lumping_adapted_system(T, y_prefix="y")
print("T* =", out["T_star"])
print("H_lumped =", out["H_lumped"])
print("Depends on extra y?", out["lumped_depends_on_extra"])
```

See `examples/example_lumping_adapted_system.py` for a runnable script.

---

### 4) Constrained lumping

In applications, you often want the reduced coordinates to include specific observables (e.g. conservation laws). The package supports building a “constrained” ansatz:

- you prescribe some rows of **T**,
- you add a chosen number of additional free rows,
- then you compute critical conditions for the resulting symbolic **T**.

```python
integrals = net.stoichiometric_first_integrals()

T, free_params = an.build_constrained_lumping_matrix(
    prescribed_rows=integrals,
    n_free_rows=1,
    free_param_prefix="p",
)

res = an.find_critical_parameters(T)
print("Free parameters:", free_params)
print("Conditions:")
for c in res["conditions"]:
    print(c, "= 0")
```

---

### 5) Proper lumping (partitions)

A **proper lumping** groups species into blocks and lumps by block-sums

```text
y_p = Σ_{j∈I_p} x_j
```

The paper (Section 5) provides an efficient **Jacobian block / column-sum criterion** for proper lumpings; the package implements it directly.

```python
import sympy as sp

res = an.proper_lumping_conditions([["S", "P"], ["E"], ["C"]])
print("T =")
sp.pprint(res["T"])
print("Conditions:")
for c in res["conditions"]:
    print("  ", sp.factor(c), "= 0")
```

You can also enumerate partitions for small networks:

```python
results = an.find_proper_lumpings(n_blocks=3, max_partitions=10_000)
for r in results:
    if r["conditions"]:
        print(r["blocks"], r["conditions"])
```

---

### 6) Enumerate candidate reductions and print a report

For exploratory work, it is often convenient to enumerate a collection of *structured* candidate reductions (generic families, proper lumpings, constrained ansätze) and print them in a compact way.

```python
from lumping_analysis import (
    LumpingAnalyzer,
    ReductionReportOptions,
    format_reduction_report,
    michaelis_menten_network,
)

net = michaelis_menten_network()
an = LumpingAnalyzer(net)

results = an.enumerate_reductions(
    include_generic=True,
    include_proper=True,
    proper_n_blocks=3,
    max_partitions=10_000,
    # Optional: discard candidates that force any of these rates to be zero.
    required_nonzero_rates=net.rate_constants,
)

print(
    format_reduction_report(
        net,
        results,
        options=ReductionReportOptions(max_conditions=12, max_relations=12),
    )
)
```

This does **not** attempt to enumerate *all* linear lumpings (which is combinatorially large); see the [Computer algebra workflows](#computer-algebra-workflows) section for the general strategy.

---

## Computer algebra workflows

The paper emphasizes that the *theory* reduces many questions to solving finitely many polynomial systems, but the practical feasibility depends on the size of the network and the algebraic-geometry tooling used.

The Python code in this repository intentionally stays “lightweight” (SymPy-only). For serious ideal decomposition / component enumeration, you will typically want to use an open-source CAS that is designed for that purpose.

### Row-echelon ansatz enumeration (general linear lumpings)

Paper Section 4 (see the row-echelon discussion leading to Proposition 5) gives a canonical way to represent general full-row-rank lumping matrices up to row operations.

Fix a target reduced dimension `e < n`.

1. Choose a set of **pivot columns** `P ⊂ {1,…,n}` with `|P| = e`.
2. Permute columns so the pivots come first.
3. Put **T** into (reduced) row-echelon form

```text
T = [ I_e  |  T̂ ]
```

where the entries of `T̂` are free symbolic parameters.

4. A convenient kernel basis is then

```text
B = [  T̂  ]
    [ -I_{n-e} ]
```

**Implementation note.** Any column basis of `ker(T)` is valid. The sign convention is irrelevant (multiplying a column by −1 does not change the kernel). The helper method

```python
ansatz = analyzer.row_echelon_ansatz(e=3, pivot_cols=[0,1,2], free_param_prefix="t")
T = ansatz["T"]
B = ansatz["B"]
```

constructs a row-echelon ansatz matrix `T` and a *polynomial* kernel basis `B` (matching the style used in the attached worksheet/notebook). See `examples/example_gpl_row_echelon_ideal_export.py` for an end-to-end run that also exports the resulting ideal to Singular.



5. Insert `T` and `B` into the critical condition

```text
T · DF(x,k) · B = 0  (for all x)
```

and extract coefficient conditions in `x`. This yields a polynomial system in:

- the rate constants `k` (or a selected subset of parameters), and
- the ansatz parameters (the entries of `T̂`).

This is the “row-echelon ansatz enumeration” approach:

- Enumerate pivot sets `P`.
- For each `P`, set up the corresponding ansatz system.
- Solve / decompose each system (or filter it) to obtain candidate lumpings and corresponding critical parameter relations.

**Practical tips.**

- Start with small `e` (e.g. `e = n-1` or `e = n-2`) and/or constrain the form of `T̂`.
- Use chemical constraints to prescribe some rows (constrained lumping) and reduce the number of free ansatz parameters.
- Use quick symbolic rank tests / rref diagnostics to discard cases that immediately force unwanted rates to be zero.

### Component enumeration and decomposition

Once you have a polynomial condition list `c_1, …, c_m` (in parameters and ansatz variables), you can form the ideal

```text
I = ⟨ c_1, …, c_m ⟩
```

and then decompose the corresponding variety into components.

Typical tasks include:

- computing minimal associated primes (component enumeration),
- primary decomposition, or
- elimination (e.g. eliminate ansatz variables to obtain conditions purely in the rate constants).

**Open-source CAS options.**

- **Singular** (specialized for polynomial ideals; used in many algebraic-geometry workflows)
- **SageMath** (can call Singular and other backends)
- **Macaulay2** (specialized for commutative algebra)

SymPy can compute Gröbner bases but does not aim to compete with specialized systems for large decompositions.

**A minimal Singular workflow (practical template).**

If you are using this Python package, you can generate such a script directly:

```python
I = analyzer.critical_ideal(T, kernel_basis=B)
print(I.to_singular_script(compute_groebner=True, primary_decomposition=False))
```

Below is a small *template* you can adapt (Singular syntax). The key ideas are:

- declare **all ansatz variables + parameters** as ring variables,
- define the ideal generated by the critical conditions,
- optionally compute a decomposition (component enumeration), and
- optionally **eliminate** ansatz variables to obtain parameter-only relations.

```singular
// Example template (edit variables + conditions)
ring R = 0,(t1,t2,t3,k1,km1,k2,km2),dp; // dp = degree reverse lex (common choice)
option(redSB);

ideal I =
  t1 + t2 - 1,
  k1 - km2,
  t3*(k2 - km1);

// 1) Gröbner basis (often useful even before decomposition)
ideal G = groebner(I);

// 2) Component enumeration / decomposition (choose one)
// list L = minAssChar(I);   // minimal associated primes
// list L = facstd(I);       // factorized standard basis components

// 3) Elimination (example: eliminate ansatz variables t1,t2,t3)
ideal I_param = eliminate(I, t1*t2*t3);
I_param;
```

How you *order variables* matters for elimination; many users place the variables
to eliminate earlier in the order, or switch to an elimination order.

If you prefer a single toolchain, SageMath can call Singular internally, so you
can keep a Python-based workflow while delegating heavy algebra to Singular.

When interpreting components, remember that reaction-network rate constants are typically constrained by:

- nonnegativity (and often strict positivity for reactions assumed present),
- application-specific “admissible parameter” restrictions.

So it is common that many algebraic components correspond to degenerate networks where some rates are forced to zero; those can be filtered out depending on your modeling intent.

---

## Mathematica / Wolfram Language workflow

A subset of the symbolic computations used in the paper — the
critical-parameter systems for the **self-replication** model of Section 6.1
and the **two-pathway enzyme** example of Section 6.2 — is performed in
Mathematica.  The reproducible artifacts are in the
[`mathematica/`](mathematica/) folder:

- `mathematica/two_pathway_enzyme_critical_variety.wl` — plain-text Wolfram
  Language script that builds the row-echelon ansatz, the polynomial right
  kernel basis, the reversible mass-action vector field, and
  prints the coefficient conditions.  Reviewable in version control;
  reproducible from the command line via

  ```bash
  wolframscript -file mathematica/two_pathway_enzyme_critical_variety.wl
  ```

  It can equally be loaded into a Mathematica kernel
  (`<< two_pathway_enzyme_critical_variety.wl`).

- `mathematica/replication_critical_variety.wl` — the counterpart for the
  self-replication model: it builds the constrained lumping ansatz
  `T(p,q,r)`, the right-kernel basis `B`, the mass-action vector field, checks
  the two prescribed first integrals, and assembles the coefficient matrix in
  the rate constants reported in Section 6.1.

The scripts include built-in conservation-law / first-integral checks.  See
`mathematica/README.md` for detailed descriptions and a guide for exporting the
resulting ideals to Singular.

The companion Singular scripts that decompose these ideals (the 92-component
`facstd` decomposition for the enzyme example and the 22-component
`minAssChar` decomposition for the self-replication model) are in the
[`singular/`](singular/) folder; see `singular/README.md`.

---


## Reproducibility artifacts

The repository includes lightweight outputs that make the submitted computations easier to check without requiring every reviewer to have the same external CAS stack installed.

- `results/replication_minAssChar_22_components.txt` contains the 22 irreducible components reported for the self-replication ideal in Section 6.1 / Appendix 8.1.
- `results/two_pathway_facstd_first5_components.txt` contains the first five components of the 92-component `facstd` output for the two-pathway enzyme ideal, as printed in Appendix 8.2.
- `results/two_pathway_elimination_minAssChar_components.txt` contains the six parameter-only elimination components discussed in Section 6.2.
- `figures/` contains the source files and generated PDFs for the manuscript figures. The Python figure scripts write outputs to `figures/generated/`; the TikZ flowchart can be rebuilt with a LaTeX installation that provides the `standalone` class.
- `requirements-repro.txt` records the Python package versions used for the final smoke test in this repository. The Section 6 ideal decompositions additionally require Singular; the Wolfram Language scripts require Mathematica/WolframScript.

From a clean checkout, the lightweight verification commands are:

```bash
pip install -e .[dev]
python -m pytest -q
python tools/verify_cas_inputs.py
```

The last command independently regenerates the Section 6 polynomial conditions with SymPy, checks the Wolfram Language model definitions, compares the generated conditions with the ideals transcribed in the Singular scripts, and verifies the frozen result-file component counts. It does not require Singular or Wolfram/Mathematica.

The external CAS decomposition scripts are intentionally not part of the automated Python test suite.

---

## Verification notes

The [`notes/`](notes/) folder contains short SymPy scripts that
independently verify selected mathematical claims of the manuscript:

- `notes/verify_ex_mm_reduced_system.py` — symbolic confirmation that with
  the labels `y_1 = s+p`, `y_2 = e+c`, `y_3 = s+p+c` used in Example `ex:mm`
  (Section 4.4), the nontrivial dynamics at the critical parameter
  `k_1 = k_{-2}` belong to `dy_1/dt`, and that `y_2` and `y_3` are first
  integrals.

- `notes/verify_section_5_formulas.py` — SymPy confirmation of the
  proper-lumping coefficient/scaling formulas of Section 5: the
  linear-coefficient scaling under variable rescaling, the column-sum
  characterisation `mu_{pq}` of the reduced linear coefficient, and the
  `-(m-1)*rho` diagonal of the intra-block correction matrix that makes
  `y_p` a first integral.

Both scripts run in a few seconds and print a step-by-step derivation
suitable for sharing.

The [`tools/`](tools/) folder contains repository-level audit scripts. The most relevant one for the external computer-algebra workflows is:

- `tools/verify_cas_inputs.py` — a SymPy-only audit that rebuilds the self-replication and two-pathway enzyme ideals from `T*DF*B = 0`, checks the corresponding Wolfram Language scripts for the expected vector fields and ansatz matrices, compares the regenerated generators with the `ideal g` declarations in the Singular scripts, verifies the frozen output component counts, and checks the explicit two-pathway Component [6] reduced system. It is a sanity check for the CAS inputs and quoted outputs; it does not independently prove that Singular’s full prime decompositions are complete or minimal.

Run it from the repository root:

```bash
python tools/verify_cas_inputs.py
```

---

## Numerical validation

For a *small* exact lumping, you can optionally attempt to construct the reduced polynomial system `y' = G(y)` (Li–Rabitz viewpoint) and compare full vs reduced simulations.

```python
import numpy as np
import sympy as sp
from lumping_analysis import michaelis_menten_network, LumpingAnalyzer

net = michaelis_menten_network()
an = LumpingAnalyzer(net)

T = sp.Matrix([
    [1, 0, 1, 1],
    [0, 1, 1, 0],
])

k1, km1, k2, km2 = net.rate_constants
kvals = {k1: 1.0, km1: 0.5, k2: 0.8, km2: 1.0}

x0 = np.array([1.0, 0.5, 0.0, 0.0])
val = an.validate_numerically(T, kvals, x0, t_span=(0.0, 10.0))
print(val)
```

For larger systems, the symbolic construction of `G` can be expensive; in that case you can still use the **condition error** reported by `validate_numerically` as a diagnostic.

---

## Troubleshooting

### “No solutions found” (or the conditions look inconsistent)

- Factor the returned conditions (`sympy.factor`) and check for obvious contradictions.
- If the resulting polynomial system is large, consider exporting it to a dedicated CAS (see above).

### “ker(T) is trivial”

Your **T** has full rank `n` (no reduction). Reduce the number of rows of **T**.

### Symbolic computations are slow

- Try smaller examples first (the built-in examples are chosen to be manageable).
- Use constrained ansätze to reduce the number of free symbolic parameters.
- Use partition-based proper lumping for quick screening.

---

## License

MIT License. See `LICENSE`.

---

## How to cite

If you use this code in research, please cite the accompanying manuscript and this software repository. A machine-readable citation template is provided in `CITATION.cff`.
