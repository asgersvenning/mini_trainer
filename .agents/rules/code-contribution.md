---
trigger: always_on
---

* Follow the `ruff` style rules; verify with: `ruff check mini_trainer`.
* Please avoid adding new dependencies 🙂 (unless **strictly** necessary!)
* Only include **necessary** comments which add crucial context that is **hard** to deduce from the code itself.
* Code should be **concise, efficient, and readable**. For example, ask yourself these questions:
  - Does it improve readability to use three lines for a for loop instead of a list comprehension?
  - Do I need this variable, or can I use it immediately?
  - Am I sacrificing efficiency and practicality for readability?
  - Should I create an abstract utility function?
  - Is this code/function/class contained in a module/submodule that makes sense?
* It is always more important to **improve** existing functionality, than to add new features.
* When altering any core machine-learning code, it is very important to ask yourself: Do I actually understand this code, why it is written in this manner, and how it interacts with the rest of the codebase? If not, don't change it.