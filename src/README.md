# Source code

Planned modules:

- `data/`: dataset adapters and validation;
- `tasks/`: leakage-resistant few-shot task construction;
- `models/`: shared model definitions;
- `meta/`: Reptile implementation;
- `baselines/`: pooled training and task-specific fine-tuning;
- `evaluation/`: metrics, uncertainty, and reporting.

Implement only against an audited dataset schema. Keep data access, task construction, model training, and evaluation independently testable.
