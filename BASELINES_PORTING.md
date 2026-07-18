# Porting the 4 review-based baselines to another Cornac branch

Four review-based baselines ported from iRev / Neu-Review-Rec:

| Baseline | Class | Notes |
|----------|-------|-------|
| **ALFM** | `cornac.models.ALFM` | Aspect-Aware Latent Factor Model. Has a Cython ext — needs building. |
| **CARP** | `cornac.models.CARP` | A Capsule Network for Recommendation. |
| **DAML** | `cornac.models.DAML` | Dual Attention Mutual Learning (a.k.a. "ADML"). |
| **MAN**  | `cornac.models.MAN`  | Main-Auxiliary Network. |

They only use stable upstream APIs, so no core file changes beyond the registration edits in section 2.

## 1. Copy these files/dirs verbatim

```
cornac/models/alfm/     # ALFM ships _alfm.pyx (Cython) — rebuild on target, see 2b
cornac/models/carp/
cornac/models/daml/
cornac/models/man/
examples/{alfm,carp,daml,man}_example.py
tests/cornac/models/test_review_models.py   # all 4, auto-skips without torch
tuning/                 # hyperparameter-tuning harness
```

## 2. Edit these 4 existing files

**a. `cornac/models/__init__.py`** — add imports (keep alphabetical):
```python
from .alfm import ALFM
from .carp import CARP
from .daml import DAML
from .man import MAN
```

**b. `setup.py`** — add to the `extensions = [...]` list, then `pip install -e .`:
```python
    Extension(
        name="cornac.models.alfm._alfm",
        sources=["cornac/models/alfm/_alfm.pyx"],
        include_dirs=[np.get_include()],
    ),
```

**c. `docs/source/api_ref/models.rst`** — add automodule entries for
`cornac.models.{man.recom_man, carp.recom_carp, daml.recom_daml, alfm.recom_alfm}`.

**d. `examples/README.md`** — add one index line per `*_example.py`.

## 3. Dependencies

```
torch>=2.0.0
gensim>=4.0.0        # CARP, DAML, MAN
scikit-learn>=1.0.0  # ALFM
```

## 4. Verify

```bash
python -c "from cornac.models import ALFM, CARP, DAML, MAN; print('ok')"
python -c "import cornac.models.alfm._alfm; print('alfm ext ok')"
pytest tests/cornac/models/test_review_models.py -v
```
