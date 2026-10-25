# data/synthetic — the training data the classifier is trained on

| File | Rows | What |
|---|---|---|
| `synthetic_1000.jsonl` | 1,000 | the full set (L0 400 / L1 176 / L2 176 / L3 248; all 19 subtypes, 6 personas, 4 scenarios) |
| `train.jsonl` | 850 | training split (stratified by subtype, no augmentation) |
| `val.jsonl` | 150 | validation split: epoch selection, temperature and thresholds (`scripts/calibrate.py`) |
| `synthetic_1000.csv` | 1,000 | the same set with a `split` column, for teammates in Excel / Google Sheets |

Written on 25 Oct 2026 by Claude (Opus 5.5) in the exact format of `scripts/generate_synthetic.py`
(`generator` field `anthropic/claude-opus-5-5`), following `data/gen_matrix.yaml` scaled to 1,000 and the
hard rule in `data/taxonomy.yaml`: no methods, means, places or amounts of self-harm. The writer never saw
`data/test_set.jsonl`. Cleaned with `scripts/clean_dedup.py` against the test set (2 rows removed as too
close), blind-relabelled with opaque ids (1,100/1,103 agreed; the 3 disagreements and 10 ambiguous rows
were dropped before trimming to 1,000), then split with `scripts/split.py --augment-rate 0`.

If the test set is rewritten, re-run the leak check and the split:

```powershell
python scripts\clean_dedup.py --inputs data\synthetic\synthetic_1000.jsonl --out data\synthetic\clean.jsonl --reports runs\full\reports
python scripts\split.py --inp data\synthetic\clean.jsonl --outdir data\synthetic --augment-rate 0
```
