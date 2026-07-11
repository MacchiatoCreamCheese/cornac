# Search-space provenance

This document records, for every value in `search_spaces.py`, **where it came from**.
Every value traces to a paper, the survey benchmark table, or the iRev framework defaults.

Legend for the **Source** column:
- **[Survey T9]** — Table 9 "Algorithm Parameter Settings" in `survey.pdf` (Article 21,
  the RARS survey), the unified benchmark grid.
- **[<Model> paper]** — the model's own paper PDF (`ALFM.pdf`, `CARP.pdf`, `DAML.pdf`,
  `MAN.txt`), with the quoted line.
- **[iRev]** — default in the iRev framework config, `iRev/config/config.py`
  (`class DefaultConfig`), which the ported Cornac constructors mirror.

All source files live at the repo root: `C:\Users\nguye\integrate\{survey,ALFM,CARP,DAML}.pdf`,
`C:\Users\nguye\integrate\MAN.txt`, `C:\Users\nguye\integrate\iRev\config\config.py`.
The grids themselves are in `C:\Users\nguye\integrate\cornac\tuning\search_spaces.py`.

---

## Primary source excerpts (verbatim, as extracted)

### Survey `survey.pdf` — Table 9 (page 21), "Algorithm Parameter Settings"
```
Dimensions of user and item representations   32
Word vector dimensions                        300
Hidden state dimensions                        32–256
Encoders technics used                        TF-IDF, Word2Vec, GloVe, FastText
Dropout rate                                  0.5
Weight decay                                  1e-3, 1e-4
Maximum document length                       500 words
Learning rate                                 1e-3, 2e-3
Number of convolutional filters               100
Window slide size                             3, 4, 5
Batch size                                    32, 64, 128
Training epochs                               100
Loss function                                 MSE
Optimizer                                     ADAM, RMSprop
Regularizers                                  L1, L2
```

### `CARP.pdf`
- p.6: "d = 300). The batch size for Musical Instruments, Beer, Office ..." (word dim 300; batch is per-dataset)
- p.7: "For CARP, window size is c = 3, the iteration number τ is set ..."
- p.7: "We set the keep probability of dropout to be 0.9 and learning rate to 0.001 for
  model training. The dimension size k is set to be **25**." (keep-prob 0.9 ⇒ dropout **0.1**;
  lr **0.001**; hidden dim **k=25**. Note: pdfplumber extracts this token as "256"; per the
  paper as read, the value is **25**.)
- p.7: "For CARP, window size is **c = 3**, the iteration number **τ** is set [to 3] ... the
  aspects for each user/item is set to be 5 (i.e., **M = 5**), and **λ is 0.5**."
- p.8: "... viewpoint/aspect. Here, we choose K = 30." (K = number of viewpoints);
  "λ controls the trade-off in the multi-task learning setting of CARP ... we fix λ to be 0.5"

### `DAML.pdf`
- p.6: "... from [0.00001, 0.00002, 0.001, 0.002]. The range of dropout ratio is ..." (learning-rate search range)
- p.6: "... the size of the sliding window is 3. In addition, the embedding di- ..."
- p.7: "the number of latent features is 32 and the size of dropout ratio is ..."
- p.7: "latent feature vector for user and item is 8, the sliding window size is
  set to 3, the dropout rate is set to 0.2, the learning rate is 0.00001, and ..."
- p.7: "the regularization parameter is tested in [0.001, 0.01, 1, 10, 100]. All ..."

### `ALFM.pdf`
- p.7: "... as f = K = 5. Notice that our model could obtain better perfor- ..." (latent f = K = 5)
- p.8: "Table 3: Comparisons of adopted methods in terms of RMSE with f = K = 5."
- p.7: "... more latent factors ..." (varied across datasets)
- Model also has A aspects and regularization coefficients (Sec. 4/5), values dataset-dependent.

### `MAN.txt` (paper text; the PDF was corrupt so plain text was supplied)
- Architecture only — hyperparameters are written **symbolically**: f (number of
  convolution filters), γ (output dim after convolution / `fc_dim`), τ (window), etc.
  **No concrete numeric settings were given in the text**, so MAN's grid is built from
  [Survey T9] and [iRev] defaults.

### iRev defaults — `iRev/config/config.py`, `class DefaultConfig`
```
seed=2023  num_epochs=10  batch_size=128  lr=2e-3  weight_decay=1e-3
loss_method='mse'  drop_out=0.5  doc_len=500  word_dim=300
filters_num=100  kernel_size=3  fc_dim=32  id_emb_size=32  num_fea=1
```
(The ported Cornac constructors `cornac/models/{carp,daml,man,alfm}/recom_*.py` use these
same defaults; DAML uses `batch_size=8`, MAN `batch_size=32`, ALFM `n_topics=32`,
`batch_size=128`.)

---

## Fixed context (held constant during search) — `search_spaces.py` `_TEXT_FIXED` / per-model `fixed`

| Setting | Value | Source |
|---|---|---|
| `embedding_size` | 300 | [Survey T9] word dim 300; [CARP paper] d=300; [iRev] `word_dim=300` |
| `max_doc_length` | 500 | [Survey T9] "Maximum document length 500 words"; [iRev] `doc_len=500` |
| `max_vocab` (ReviewModality) | 50000 | [iRev] `MAX_VOCAB=50000` (`iRev/preprocess_data/constants.py`) |
| `max_doc_freq` | 0.7 | [iRev] `MAX_DF=0.7` (same file) |
| `pretrained_w2v_type` | word2vec | [Survey T9] lists Word2Vec; paper baselines use GoogleNews word2vec |
| search `max_iter` | CLI `--epochs` (default 10) | runtime budget; [iRev] `num_epochs=10`, survey states 100 |
| DAML `kernel_size` | 3 (fixed) | [DAML paper] "sliding window size is set to 3" |
| ALFM `lda_max_iter` | 10 | LDA solver iteration budget (implementation, not a modeling hyperparameter) |

---

Every grid value has a documented source. Where a parameter has only a single sourced
value, it is not swept — it is **pinned** at that value.

Each model table lists **every** hyperparameter, with its **Role**:
- **SWEPT** — varied by coordinate descent (`SPACES[<Model>]["order"]`).
- **PINNED** — single sourced value, held fixed (`SPACES[<Model>]["fixed"]` or constructor default).
- **CONTEXT** — benchmark-wide fixed setting (`_TEXT_FIXED` / `ReviewModality` args).
- **RUNTIME** — a training budget, not a modeling value.
- **INFRA** — bookkeeping (not a hyperparameter).

## CARP — every parameter

| Parameter | Role | Value / grid | Source |
|---|---|---|---|
| learning_rate | SWEPT | 1e-3, 2e-3 | 1e-3 = [CARP paper] p.7 "learning rate to 0.001"; 2e-3 = [Survey T9]/[iRev] |
| dropout_rate | SWEPT | 0.1, 0.5 | 0.1 = [CARP paper] (paper reports **keep-prob 0.9** = drop 0.1); 0.5 = [Survey T9]/[iRev] |
| weight_decay | SWEPT | 1e-4, 1e-3 | [Survey T9]; 1e-3 also [iRev] |
| kernel_size | PINNED | 3 | [CARP paper] "window size c = 3"; also [iRev] `kernel_size=3` |
| batch_size | PINNED | 128 | [iRev] `batch_size=128` (DefaultConfig; no CARP override in `train.sh`) |
| id_embedding_size | PINNED | 32 | [iRev] `id_emb_size=32` / [Survey T9]. (Paper's `k=25` is NOT used — iRev used 32.) |
| n_filters | PINNED | 100 | [Survey T9]/[iRev] `filters_num=100` |
| embedding_size | CONTEXT | 300 | [Survey T9] word dim; [CARP paper] d=300; [iRev] |
| max_doc_length | CONTEXT | 500 | [Survey T9]; [iRev] `doc_len` |
| max_vocab | CONTEXT | 50000 | [iRev] `MAX_VOCAB` |
| max_doc_freq | CONTEXT | 0.7 | [iRev] `MAX_DF` |
| pretrained_w2v_type | CONTEXT | word2vec | [Survey T9] lists Word2Vec; GoogleNews baseline |
| pretrained_w2v_path | CONTEXT | GoogleNews file | word2vec embedding source (setup) |
| max_iter | RUNTIME | `--epochs` (default 10) | training budget; [iRev] `num_epochs=10`, survey states 100 |
| name / trainable / verbose / seed / init_params | INFRA | — | not tuned |

**CARP paper knobs that are NOT modeled — absent from iRev too** (verified in
`iRev/models/carp.py`):

| Paper knob | Paper value | iRev | Our port |
|---|---|---|---|
| `k` (dimension) | 25 | uses `id_emb_size=32` instead | pinned `id_embedding_size=32` (iRev-faithful) |
| `c` (window) | 3 | `kernel_size=3` | pinned `kernel_size=3` ✅ |
| `τ` (routing iterations) | 3 | not implemented (single coupling step; dead `routing` scalar) | not implemented |
| `λ` (multi-task loss weight) | 0.5 | not implemented (shared LFM head + MSE) | not implemented |
| `M` (aspects / user-item) | 5 | not implemented | not implemented |
| `K` (viewpoints) | 30 | not exposed | not exposed |

iRev deliberately fit CARP into its common framework (shared LFM head, MSE loss,
`id_emb=32`), dropping the paper's dynamic routing (`τ`), multi-task objective (`λ`),
`M`-aspect structure, and `k`. Our port mirrors iRev — so reproducing the **benchmark**
means these stay unmodeled; matching the **original paper** would require re-implementing
them.

## DAML — every parameter

| Parameter | Role | Value / grid | Source |
|---|---|---|---|
| learning_rate | SWEPT | 1e-3, 2e-3 | [Survey T9] (subset of [DAML paper] range; 1e-5/2e-5 excluded — see open items) |
| dropout_rate | SWEPT | 0.2, 0.5 | 0.2 = [DAML paper]; 0.5 = [Survey T9]/[iRev] |
| id_embedding_size | SWEPT | 8, 32 | 8 = [DAML paper] ("vector for user and item is 8"); 32 = [DAML paper]/[Survey T9] |
| weight_decay | SWEPT | 1e-4, 1e-3 | [Survey T9] (paper's [0.001…100] is a different reg term — see open items) |
| batch_size | SWEPT | 8, 32, 64, 128 | 8 = [iRev] default; 32/64/128 = [Survey T9] (large may OOM at doc_len 500) |
| kernel_size | PINNED | 3 | [DAML paper] "sliding window size is set to 3" |
| n_filters | PINNED | 100 | [Survey T9] |
| embedding_size | CONTEXT | 300 | [Survey T9]; [iRev] |
| max_doc_length | CONTEXT | 500 | [Survey T9]; [iRev] |
| max_vocab | CONTEXT | 50000 | [iRev] `MAX_VOCAB` |
| max_doc_freq | CONTEXT | 0.7 | [iRev] `MAX_DF` |
| pretrained_w2v_type | CONTEXT | word2vec | [Survey T9]; GoogleNews baseline |
| pretrained_w2v_path | CONTEXT | GoogleNews file | word2vec source (setup) |
| max_iter | RUNTIME | `--epochs` (default 10) | training budget; [iRev] `num_epochs=10`, survey states 100 |
| name / trainable / verbose / seed / init_params | INFRA | — | not tuned |

## MAN — every parameter

The MAN paper (`MAN.txt`) gives **no numeric hyperparameters**, so nothing here is
paper-sourced beyond [Survey T9]/[iRev].

| Parameter | Role | Value / grid | Source |
|---|---|---|---|
| learning_rate | SWEPT | 1e-3, 2e-3 | [Survey T9]; 2e-3 also [iRev] |
| weight_decay | SWEPT | 1e-4, 1e-3 | [Survey T9] |
| kernel_size | PINNED | 3 | [Survey T9] window; 3 also [iRev]. Pinned (sliding window = 3). |
| batch_size | PINNED | 128 | pinned per request. NOTE: [iRev] `train.sh` ran MAN with **batch_size=32**, not 128 |
| dropout_rate | PINNED | 0.5 | [Survey T9]/[iRev] |
| id_embedding_size | PINNED | 32 | [Survey T9]/[iRev] |
| n_filters | PINNED | 100 | [Survey T9]/[iRev] |
| fc_dim (γ) | PINNED | 32 | [iRev] `fc_dim` |
| embedding_size | CONTEXT | 300 | [Survey T9]; [iRev] |
| max_doc_length | CONTEXT | 500 | [Survey T9]; [iRev] |
| max_vocab | CONTEXT | 50000 | [iRev] `MAX_VOCAB` |
| max_doc_freq | CONTEXT | 0.7 | [iRev] `MAX_DF` |
| pretrained_w2v_type | CONTEXT | word2vec | [Survey T9]; GoogleNews baseline |
| pretrained_w2v_path | CONTEXT | GoogleNews file | word2vec source (setup) |
| max_iter | RUNTIME | `--epochs` (default 10) | training budget; [iRev] `num_epochs=10`, survey states 100 |
| name / trainable / verbose / seed / init_params | INFRA | — | not tuned |

## ALFM — every parameter

Topic model — **no word embeddings** (no `n_filters`, `kernel_size`, `embedding_size`,
`pretrained_w2v_*`). The paper's `f=K=5` is the latent-factor dim (≈ id embedding), not our
LDA `n_topics`; its aspect count `A` was not stated numerically — so neither is used.

| Parameter | Role | Value / grid | Source |
|---|---|---|---|
| learning_rate | SWEPT | 1e-3, 2e-3 | [Survey T9]; 2e-3 also [iRev] |
| weight_decay | SWEPT | 1e-4, 1e-3 | [Survey T9] |
| batch_size | SWEPT | 32, 64, 128 | [Survey T9]; 128 also [iRev] |
| dropout_rate | PINNED | 0.5 | [Survey T9]/[iRev] (not an original-ALFM knob) |
| n_topics | PINNED | 32 | [iRev] LDA-32 topic matrix |
| id_embedding_size | PINNED | 32 | [Survey T9]/[iRev] |
| max_doc_length | CONTEXT | 500 | [Survey T9]; [iRev] |
| max_vocab | CONTEXT | 50000 | [iRev] `MAX_VOCAB` |
| max_doc_freq | CONTEXT | 0.7 | [iRev] `MAX_DF` |
| lda_max_iter | RUNTIME | 10 | LDA solver iteration budget (implementation) |
| max_iter | RUNTIME | `--epochs` (default 10) | training budget; [iRev] `num_epochs=10`, survey states 100 |
| name / trainable / verbose / seed / init_params | INFRA | — | not tuned |

---

## Summary

Every grid value traces to **[Survey T9]**, a **model paper**, or **[iRev]**. Parameters
whose only sourced value equals the default are pinned (not swept): CARP
`kernel/id/filters/batch`; DAML `n_filters` (+`kernel`); MAN
`dropout/id/filters/fc_dim/kernel/batch`; ALFM `dropout/n_topics/id`.

**Paper values not used in the grid** (deliberate, for reproducing the iRev/survey benchmark
rather than each original paper):
1. **CARP paper knobs** — `k=25` (dim), `τ=3` (routing iters), `λ=0.5` (multi-task loss),
   `M=5` (aspects) are **not modeled** — and were **not in iRev either** (iRev used
   `id_emb=32`, shared LFM head + MSE, single-step coupling). Matching the paper needs
   re-implementing routing/multi-task/aspects.
2. **DAML learning_rate** — the paper's `1e-5`/`2e-5` are **not** in the grid (they need
   many more epochs).
3. **DAML regularization** — uses survey `weight_decay {1e-4,1e-3}`, not the paper's
   `[0.001, 0.01, 1, 10, 100]` reg sweep (a different regularization term).
4. **ALFM** — paper's `f=K=5` (latent dim) and aspect count `A` are not swept.
5. **MAN** — no paper hyperparameters exist; grid is survey/iRev only.
6. **Epochs** — `--epochs` is a runtime budget, not a modeling value from the survey's 100.

To use any of these, edit `SPACES[...]` in `search_spaces.py`.

## How to re-verify the source excerpts

The PDF lines above were extracted with `pdfplumber` (survey/ALFM/CARP/DAML) and read
directly from `MAN.txt`. To reprint any page:
```python
import pdfplumber
with pdfplumber.open("CARP.pdf") as pdf:
    print(pdf.pages[6].extract_text())   # 0-indexed -> page 7
```
