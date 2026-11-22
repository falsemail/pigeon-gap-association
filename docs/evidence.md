# Evidence and units

Each file under `data/decisions/` contains a single external recording. Each line is one registered perturbation of a pair of visible windows. The fields `window_index` and `condition_index` identify the condition in `data/frozen-registry.json`. The field `registry` is the SHA-256 of the original registry. Relative paths in that registry identify files in the historical experiment.

| Field | Meaning |
|---|---|
| `gap`, `sigma` | Removed interval in seconds and added position-noise scale in metres |
| `turnover`, `clutter` | Imposed membership turnover and added clutter, expressed as fractions |
| `truth_a`, `truth_b` | Local reference identities for each tracklet before and after the gap. Negative labels denote clutter |
| `prediction` | One successor index per earlier tracklet. −1 means unmatched |
| `retention` | Correct links divided by the number of real identities visible on both sides |
| `precision` | Correct links divided by the number of accepted links |
| `f1` | Twice the correct-link count divided by the number of continuing identities plus the number of accepted links |
| `union_count_error` | Absolute difference between the estimated and reference counts of distinct real individuals across both windows |
| `fallbacks` | Registered fallback events for a particular method and condition |

A denominator of zero gives a score of zero for the corresponding ratio. The external evaluation averages conditions within a recording and then averages recordings within a source and gap. Retention, precision and F1 are stored as proportions in the full-precision outputs. The display tables for the main text report percentages. Paired contrasts use percentage points. Intervals resample recordings and describe variation among the available recordings. Individuals can appear in more than one recording.

Development Table S9 uses a separate set of 52 recording-by-gap rows, 336 conditions at 15 s and 351 at 20 s. The two threshold columns in each record share their inputs. For overall estimates the scorer pools durations within each recording using the condition counts and then averages recordings within sources and equally across the three sources. Unlike the display in Supplementary Table S9, the CSV stores F1, precision and retention as proportions. Count errors are in individuals.

The original external sources, original archive checksums and access routes are listed in [the source manifest](../data/source-provenance.json). Archived computational timing measurements in `data/cost-extension.csv` depend on the original hardware. They are not re-measured by the scoring scripts.
