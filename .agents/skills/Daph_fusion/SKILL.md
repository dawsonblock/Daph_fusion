```markdown
# Daph_fusion Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you how to contribute to the Daph_fusion Python codebase, which focuses on task arithmetic operators and model merging logic. You'll learn the project's coding conventions, commit patterns, testing strategies, and step-by-step workflows for adding or correcting task arithmetic operators, including how to route them through the pipeline, export them, and validate them with tests and CI.

## Coding Conventions

- **File Naming:**  
  Use `snake_case` for all Python files and modules.  
  *Example:*  
  ```
  daph_exfusion/merge/task_arithmetic.py
  tests/test_v3_corrected_dense_merge.py
  ```

- **Import Style:**  
  Use **relative imports** within packages.  
  *Example:*  
  ```python
  from .fisher_task_arithmetic import FisherTaskArithmetic
  ```

- **Export Style:**  
  Use **named exports** in `__init__.py` to expose specific classes or functions.  
  *Example:*  
  ```python
  from .task_arithmetic import TaskArithmetic
  from .fisher_task_arithmetic import FisherTaskArithmetic

  __all__ = ["TaskArithmetic", "FisherTaskArithmetic"]
  ```

- **Commit Messages:**  
  Follow the **conventional commit** format with prefixes like `fix`, `feat`, `chore`, `test`, `ci`.  
  *Example:*  
  ```
  feat: add corrected dense merge operator
  fix: resolve bug in fisher task arithmetic
  ```

## Workflows

### Add or Correct Task Arithmetic Operator
**Trigger:** When you need to implement or fix a task arithmetic operator and integrate it into the system.  
**Command:** `/add-task-arithmetic-operator`

1. **Implement or fix the operator logic**  
   - Add or update the operator in a file under `daph_exfusion/merge/`, such as `task_arithmetic.py` or `fisher_task_arithmetic.py`.  
   - Example:
     ```python
     class NewTaskArithmetic:
         def merge(self, models):
             # Operator logic here
             pass
     ```
2. **Update the pipeline**  
   - Modify `pipeline_v3.py` to use the new or corrected operator.
   - Example:
     ```python
     from .task_arithmetic import NewTaskArithmetic

     def run_pipeline(...):
         operator = NewTaskArithmetic()
         ...
     ```
3. **Export the operator**  
   - Add the new operator to `daph_exfusion/merge/__init__.py` for package-level access.
   - Example:
     ```python
     from .new_task_arithmetic import NewTaskArithmetic

     __all__ = ["TaskArithmetic", "FisherTaskArithmetic", "NewTaskArithmetic"]
     ```
4. **Add or update tests**  
   - Create or modify test files in `tests/`, following the `*.test.*` pattern.
   - Example:
     ```python
     # tests/test_v3_corrected_dense_merge.py
     def test_new_task_arithmetic_merge():
         ...
     ```
5. **Update or add CI workflow**  
   - Ensure `.github/workflows/exfusion-v3-corrected.yml` validates and packages the new operator.

## Testing Patterns

- **Test File Naming:**  
  Test files follow the `*.test.*` pattern and are placed in the `tests/` directory.  
  *Example:*  
  ```
  tests/test_v3_corrected_dense_merge.py
  ```

- **Testing Framework:**  
  The specific framework is not detected, but tests are written as Python functions, likely using `pytest` or similar.

- **Test Example:**  
  ```python
  def test_new_task_arithmetic_merge():
      operator = NewTaskArithmetic()
      result = operator.merge([model1, model2])
      assert result is not None
  ```

## Commands

| Command                         | Purpose                                                           |
|----------------------------------|-------------------------------------------------------------------|
| /add-task-arithmetic-operator    | Scaffold and integrate a new or corrected task arithmetic operator |
```
