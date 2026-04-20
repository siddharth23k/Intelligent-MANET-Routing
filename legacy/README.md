# Legacy Directory - ARCHIVED

**Status: DEPRECATED - DO NOT USE**

This directory contains the original implementation that has been superseded by the new clean structure.

## What's Here

All files in this directory are **archived** and **non-functional** due to missing dependencies and outdated paths:

### Broken Data Processing
- `build_dataset.py` - Superseded by `pipeline/generate_data.py`
- `feature_engineering.py` - Superseded by `pipeline/engineer_features.py`
- `add_failure_label.py` - Logic integrated into new pipeline

### Broken Evaluation Scripts
- `evaluate_ours.py` - Functionality moved to `pipeline/compare_methods.py`
- `evaluate_classic_baseline.py` - Functionality moved to `pipeline/compare_methods.py`
- `evaluate_paper_baseline.py` - Functionality moved to `pipeline/compare_methods.py`
- `evaluate_routing.py` - Superseded by comparison framework

## Why This Doesn't Work

1. **Missing Directories**: References non-existent `src/` and `dataset/` directories
2. **Path Inconsistencies**: Uses old `dataset/` paths instead of current `data/` paths
3. **Broken Imports**: All import paths are invalid in the new structure
4. **Duplicate Functionality**: All functionality has been reimplemented in the new pipeline

## Current Working Structure

Use the new structure instead:

```
Intelligent-MANET-Routing/
|-- methods/           # Current method implementations
|-- pipeline/          # Data processing & training pipeline
|-- data/              # Current data location
|-- simulation/        # NS-3 simulation setup
|-- results/           # Results and models
|-- config/            # Configuration files
```

## Migration Guide

| Old Legacy File | New Replacement |
|------------------|-----------------|
| `legacy/build_dataset.py` | `pipeline/generate_data.py` |
| `legacy/feature_engineering.py` | `pipeline/engineer_features.py` |
| `legacy/evaluate_*.py` | `pipeline/compare_methods.py` |
| `dataset/` paths | `data/` paths |

## Recommendation

**DO NOT** use any files in this directory. They are kept for historical reference only.
All active development should use the new pipeline structure.

If you need to understand the original implementation approach, you can reference these files, but do not attempt to run them.
