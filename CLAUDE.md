Bible: Less is more. Simplicity is best. Always opt for simplicity and excellent documentation. 

# Debugging Guide 
1. reflect on 5-7 different possible sources of the problem, distill those down to 1-2 most likely sources, and then add logs to validate your assumptions before we move onto implementing the actual code fix. 

# General Principles for Clean, Readable, Scalable, Robust ML Code
## 1. Code Style & Formatting
- **Consistent Formatting**  
  Use a linter/formatter (e.g. Black, isort, flake8) with a project-wide configuration.
- **Clear Naming**  
  - Variables/functions: `snake_case`  
  - Classes: `PascalCase`  
  - Avoid single-letter names except in tight loops (e.g. `i`, `j`).
- **No “Fun” Symbols**  
  Under no circumstances should you use emojis or other non-standard characters—strip them out if found.

## 2. Modularity & Organization
- **Single Responsibility**  
  Each function or class should have one clear purpose. If it grows beyond ~20 lines or covers more than one concern, split it.
- **Layered Structure**  
  1. **Data layer**: loading, preprocessing, augmentation  
  2. **Model layer**: architecture definitions  
  3. **Train/Eval layer**: training loops, metrics logging  
  4. **Utils/Common**: shared helpers (e.g. metrics, plotting)
- **Config-Driven**  
  Centralize hyperparameters, paths, and flags in a single config file (YAML/JSON) rather than hard-coding.

## 3. Robustness & Safety
- **Defensive Checks**  
  - Use `assert` (or custom exceptions) for input shapes, dtypes, value ranges.  
  - Validate config values at startup and fail fast with clear messages.
- **Error Handling**  
  Use `try/except` sparingly around I/O or external calls; avoid bare `except:`.
- **Reproducibility**  
  - Fix random seeds for all frameworks (NumPy, PyTorch, TensorFlow, Python’s `random`).  
  - Log dependency versions (e.g. via `pip freeze`) alongside experiment outputs.

## 4. Testing & Validation
- **Unit Tests**  
  Cover core transforms, loss functions, and custom math logic with pytest (or unittest).
- **Integration Tests**  
  Smoke-test end-to-end training on tiny datasets (e.g. 10 samples).
- **Continuous Integration**  
  Hook tests into CI (GitHub Actions, GitLab CI) to catch regressions on each push.

## 5. Documentation & Readability
- **Docstrings & Type Hints**  
  - Public functions/classes get docstrings (Google or NumPy style).  
  - Use Python type hints:  
    ```python
    def forward(x: Tensor) -> Tensor:
        ...
    ```
- **README & Tutorials**  
  Provide a project-root README that explains setup, data download, quickstart, and how to reproduce key experiments.
- **In-Code Comments**  
  Explain “why” a non-trivial step exists, not “what” (the code should speak for itself).

## 6. Experiment Tracking & Logging
- **Structured Logging**  
  Use a logging framework (`logging`, Weights & Biases, TensorBoard) instead of `print`.
- **Checkpointing**  
  - Save model checkpoints regularly with descriptive filenames (e.g. `epoch_10-valacc_0.82.pt`).  
  - Maintain a “latest” symlink or pointer for easy resume.
- **Metadata**  
  Save config files, random seeds, Git commit hash, and environment info alongside results.

## 7. Dependency Management & Environments
- **Isolated Environments**  
  Use venv, conda, or Poetry to isolate dependencies.
- **Pinned Requirements**  
  Commit a `requirements.txt` or lockfile so collaborators use the same versions.

## 8. Performance & Scalability
- **Vectorization**  
  Favor batch operations over Python loops.
- **Profiling**  
  Periodically profile hotspots (e.g. with `cProfile` or PyTorch’s profiler) and document findings.
- **Lazy/Data Streaming**  
  For large datasets, use streaming or memory-mapping rather than loading everything into RAM.

## 9. Collaboration & Code Review
- **Git Workflow**  
  - Feature branches per experiment or feature.  
  - Descriptive commit messages (e.g. “Add cosine warmup scheduler”).
- **Pull Requests**  
  Require at least one reviewer; enforce linting and tests in CI.

---

