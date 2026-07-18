## Shared context (`_TEXT_FIXED` / ReviewModality)


| Setting               | Value          | Source                                                                                           |
| --------------------- | -------------- | ------------------------------------------------------------------------------------------------ |
| `embedding_size`      | 300            | [Survey T9]; [CARP] d=300; [iRev] `word_dim=300`                                                 |
| `max_doc_length`      | 500            | [Survey T9]; [iRev] `doc_len=500` — overridden: CARP **300** [official], MAN **1000** [MAN §5.4] |
| `max_vocab`           | 50000          | [iRev] `MAX_VOCAB`                                                                               |
| `max_doc_freq`        | 0.7            | [iRev] `MAX_DF`                                                                                  |
| `pretrained_w2v_type` | word2vec       | [Survey T9]; GoogleNews baseline                                                                 |
| `max_iter`            | CLI `--epochs` | runtime budget                                                                                   |


ALFM ignores embeddings / doc length.

---

## CARP

Official [https://github.com/WHUIR/CARP](https://github.com/WHUIR/CARP). Sweeps paper ablations; rest pinned to paper/runner.


| Parameter           | Role   | Value / grid            | Source                            |
| ------------------- | ------ | ----------------------- | --------------------------------- |
| `num_aspect` (M)    | SWEPT  | 5, 3, 7, 9              | [paper] Table 3                   |
| `gama` (λ)          | SWEPT  | 0.5, 0.1, 0.3, 0.7, 1.0 | [paper] Fig 3(b); default 0.5     |
| `itr_routing` (τ)   | SWEPT  | 3, 1, 2, 4              | [paper] Table 4                   |
| `latent_dim` (k)    | SWEPT  | 25, 50, 100             | [paper] k=25; optimal in [25,100] |
| `learning_rate`     | PINNED | 1e-3                    | [paper] p.7                       |
| `dropout_keep_prob` | PINNED | 0.9                     | [paper] p.7                       |
| `n_filters`         | PINNED | 50                      | [official] `num_filters=50`       |
| `kernel_size`       | PINNED | 3                       | [paper] c=3                       |
| `itr_self_attn`     | PINNED | 2                       | [official] `itr_0=2`              |
| `lambda_1`          | PINNED | 0.8                     | [official]                        |
| `rating_threshold`  | PINNED | 3.0                     | [official]                        |
| `batch_size`        | PINNED | 100                     | [official]                        |


---



## DAML

NFM + add fusion via [NRR]. Grid from [paper] p.6–7 + [NRR].

Omitted from paper ranges: lr {1e-5, 2e-5} (epoch budget); reg {1, 10, 100} (Adam WD).


| Parameter           | Role   | Value / grid            | Source                                    |
| ------------------- | ------ | ----------------------- | ----------------------------------------- |
| `learning_rate`     | SWEPT  | 2e-3, 1e-3              | [paper] {1e-5,2e-5,1e-3,2e-3}; [NRR] 2e-3 |
| `dropout_rate`      | SWEPT  | 0.5, 0.2, 0.1, 0.3, 0.4 | [paper] {0.1..0.4}; [NRR] 0.5             |
| `id_embedding_size` | SWEPT  | 32, 8, 16, 64, 128      | [paper] {8..128}; [NRR] 32                |
| `weight_decay`      | SWEPT  | 1e-3, 1e-2              | [paper] {0.001,0.01}; [NRR] 1e-3          |
| `batch_size`        | SWEPT  | 8, 32, 64, 128          | [NRR] 128; smaller for O(doc_len²) memory |
| `kernel_size`       | PINNED | 3                       | [paper]; [NRR]                            |
| `n_filters`         | PINNED | 100                     | [paper]; [NRR]                            |


---



## MAN

From [MAN §5.4]. Omitted: lr {1e-5, 6e-5}. Embeddings stay 300-dim (paper: 64-dim).


| Parameter            | Role   | Value / grid            | Source                                      |
| -------------------- | ------ | ----------------------- | ------------------------------------------- |
| `learning_rate`      | SWEPT  | 6e-3, 1e-3              | [§5.4] {1e-5,6e-5,1e-3,6e-3}; MAN uses 6e-3 |
| `dropout_rate`       | SWEPT  | 0.5, 0.1, 0.2, 0.3, 0.4 | [§5.4] {0.1..0.5}                           |
| `id_embedding_size`  | SWEPT  | 32, 8, 16, 64           | [§5.4]                                      |
| `n_filters`          | SWEPT  | 100, 50, 150, 200       | [§5.4]                                      |
| `weight_decay`       | PINNED | 1e-3                    | [§5.4] λ=10⁻³                               |
| `batch_size`         | PINNED | 128                     | [§5.4]                                      |
| `fc_dim`             | PINNED | 50                      | [§5.4]                                      |
| `n_heads` / `ff_dim` | PINNED | 4 / 128                 | [§5.4]                                      |
| `att_hidden`         | PINNED | 64                      | [§5.4]                                      |
| `kernel_size`        | PINNED | 3                       | [§5.4]                                      |
| `max_rt_length`      | PINNED | 50                      | [§5.4]                                      |


---



## ALFM

Official Java ATM + SGD. Sweeps f, K (Fig 2/3); rest [official].


| Parameter                                | Role   | Value / grid      | Source                |
| ---------------------------------------- | ------ | ----------------- | --------------------- |
| `num_factors` (f)                        | SWEPT  | 5, 10, 15, 20, 25 | [paper] Fig 2/3       |
| `n_topics` (K)                           | SWEPT  | 5, 10, 15, 20, 25 | [paper] Fig 2/3       |
| `num_aspects`                            | PINNED | 5                 | [official]            |
| `alpha` / `beta` / `gamma`               | PINNED | 0.1 / 0.01 / 0.5  | [official] ATM priors |
| `tm_iterations` / begin_save / save_step | PINNED | 120 / 100 / 10    | [official]            |
| `learn_rate`                             | PINNED | 0.01              | [official]            |
| `reg`                                    | PINNED | 0.5               | [official]            |
| `weight_reg`                             | PINNED | 0.01              | [official]            |


