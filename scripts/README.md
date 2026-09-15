# Script organization

Production logic lives in `src/plan_c/`. The `scripts/` tree contains entry
points and optional tools organized by why they are run.

| Folder | Purpose | Production path? |
|---|---|---|
| `training/` | Legacy final-model training command | Use the canonical command instead |
| `evaluation/` | LORO, comparison, and transfer evaluation | Evaluation only |
| `experiments/` | Ablations, feature selection, detector/window comparisons | No |
| `diagnostics/` | Data-quality and error investigation | No |
| `plots/` | Visualize data or stored results | No |
| `exports/` | Export intermediate values to CSV, Excel, or JSON | No |
| `data_sources/` | Obtain raw recordings | Before the main workflow |

The canonical production workflow is exposed through one entry point:

```powershell
python -m src.plan_c run-all --output-dir outputs/plan_c
```

See the README inside each folder for its individual commands. Run commands
from the repository root so relative input and output paths stay predictable.
