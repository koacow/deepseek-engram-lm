# Run Pipeline

## 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 2. Train the baseline model

```bash
python scripts/train_baseline.py
```

## 3. Train the parameter-matched control

```bash
python scripts/train_params_control.py
```

## 4. Train the full Engram model

```bash
python scripts/train_engram.py
```

## 5. Train the frozen-gate Engram ablation

```bash
python scripts/train_engram.py --frozen-gates
```

## 6. Example of a longer run

```bash
python scripts/train_engram.py --steps 5000 --batch-size 8 --output-dir checkpoints
```

## 7. Notes

- Scripts assume the repository root as the working directory.
- Checkpoints are written to the directory passed with `--output-dir`.
- The Engram script defaults to the learned-gate variant unless `--frozen-gates` is set.
