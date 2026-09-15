# Manual exports

These utilities are for inspecting intermediate signals and features. They are
not part of the production `src.plan_c` workflow.

Run them from the repository root:

```powershell
python scripts/exports/extract_features_to_excel.py data/<recording>.txt
python scripts/exports/export_rr_rrv_excel.py --input <input.xlsx> --output <output.xlsx>
python scripts/exports/dump_preprocessed_nn.py
python scripts/exports/get_rmssd.py
```

Default generated files continue to go to the repository-level `outputs/`
directory. Moving these scripts under `exports/` does not change feature
calculation or output contents.
