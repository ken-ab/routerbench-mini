# V5 Reproducibility

```bash
../.venv-routerbench-mini-py/bin/python scripts/build_v5_data.py --force
../.venv-routerbench-mini-py/bin/python scripts/run_v5_phase1.py --workers 16
../.venv-routerbench-mini-py/bin/python scripts/run_v5_phase2.py --workers 16
../.venv-routerbench-mini-py/bin/python -m pytest -q
```

## Modified Files

- ` M .gitignore`
- ` M src/routerbench_mini/calibration.py`
- ` M src/routerbench_mini/providers.py`
- ` M src/routerbench_mini/selection.py`
- ` M src/routerbench_mini/tasks.py`
- `?? configs/models.qwen_v5.yaml`
- `?? configs/v5_large_scale.yaml`
- `?? docs/v5_large_scale_audit.zh-CN.md`
- `?? scripts/build_v5_data.py`
- `?? scripts/run_v5_phase1.py`
- `?? scripts/run_v5_phase2.py`
- `?? src/routerbench_mini/v5.py`
- `?? tests/test_v5_protocol.py`

## Unresolved

- API backends may change implementation behind the same model alias; response metadata and run timestamps are retained.
- Exact-match and task-specific automatic graders can under-credit semantically equivalent open answers.
- V5 does not perform the intentionally excluded 300/800/1600/3200 training-size scaling study.
