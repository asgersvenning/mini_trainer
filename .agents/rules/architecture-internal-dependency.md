---
trigger: always_on
---

# **mini_trainer Architecture and Dependency Enforcement**

* **Zero Dynamic Execution:** Never run dynamic test suites (like Pytest) to validate architecture, as this initializes CUDA contexts and risks crashing the uv environment. Rely exclusively on static AST parsing.
* **Macro-Architecture (DAG & Layers):** The codebase must remain a Directed Acyclic Graph (DAG). You must read pyproject.toml [tool.importlinter.contracts] before modifying files.
  - Ensure compliance with the layers contract: utils is the bedrock and must never import upward from modeling, training, or visualization.
  - Ensure compliance with the acyclic_siblings contract. If your changes introduce a dependency cycle between sibling files, extract the shared logic into a new, independent sibling node.
* **Micro-Architecture (Import Syntax):**
  - *Sibling Files:* Co-located files in the same directory must exclusively use single-dot relative imports (e.g., from. import tensor_ops).
  - *Cross-Directory Files:* Any import crossing a directory boundary must be absolute (e.g., from mini_trainer.utils import tensor_ops).
  - *Banned Syntax:* Never use parent-level relative traversals (e.g., from..utils import tensor_ops). This is strictly banned by our Ruff TID252 configuration.
* **Validation Gate:** Before completing any task, you must successfully execute uv run ruff check. and uv run lint-imports.