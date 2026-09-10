# Worktree coordination for agents

Read this before worktree setup or concurrent agent work. Root
[AGENTS.md](../../AGENTS.md) takes precedence. Use the shared commands in
[dev/worktrees.md](../../dev/worktrees.md); do not build a second setup workflow.

## Assignment and ownership

- The coordinating agent records each active task's owner, absolute worktree
  path, branch, base commit, allowed paths, expected deliverable and validation
  boundary in its task assignment. Use existing agent messaging/task tracking;
  do not invent a committed lock file or session registry.
- Each concurrent implementation task gets a separate branch and worktree.
  Read-only review can inspect an existing worktree, but must account for files
  changing while its owner works. Review a committed revision for stable evidence.
- Agree ownership of shared interfaces before splitting dependent work. Give
  overlapping file changes to one owner or execute them sequentially. Worktrees
  prevent accidental shared-file edits; they do not prevent semantic conflicts.
- A task assignment does not authorize unrelated cleanup, changing other agents'
  scopes or starting additional agents. Follow the session's delegation rules.

## Establish the checkout before working

1. Inspect `git worktree list`, the assigned checkout's `git status --short`,
   `git branch --show-current` and `git rev-parse HEAD`. Check that the branch and
   base agree with the assignment. Preserve any existing changes.
2. Read that checkout's `AGENTS.md` and task-relevant guidance. Different branches
   can have different code and rules; the main checkout is not authoritative for
   a feature branch's implementation.
3. Use the assigned absolute path as `workdir` for every shell tool call. Use
   absolute paths rooted there for file edits. Pass the path and branch in every
   delegated task. A shell `cd` does not change future tool calls or other agents.
4. Before tests, verify imports resolve to the assigned source tree using the
   documented environment check. Set `PYTHONPATH` to that tree for subprocesses.
   A test against the main checkout is not validation of the feature worktree.

## Shared resources are still shared

- A shared `.venv` is for read-only dependency reuse. Do not install, uninstall,
  synchronize or replace packages from any participating worktree while it is in
  use. Use a separate environment for dependency changes, explicitly choosing the
  intended PyTorch backend. Do not claim the shared environment matches a branch's
  lockfile without checking it.
- Give each task its own ignored report, benchmark, log and temporary paths. Use
  existing datasets/checkpoints read-only by absolute path; do not mutate them or
  silently substitute another run's outputs.
- Coordinate GPU and large CPU/RAM jobs. Do not start competing performance runs
  or report contended measurements as comparable uncontended benchmarks. Record
  hardware, revision and environment for performance evidence.
- Git refs, configuration, object storage and some metadata are shared. Do not
  reset/rebase another task's branch, run repository-wide cleanup, remove another
  worktree, or change common Git settings without coordinating with its owner.
  Do not force worktree removal or delete another agent's Git lock files.

## Commit and hand off

- Keep implementation commits on the assigned branch. Stage explicit owned paths;
  inspect staged content and run the relevant repository checks. Keep `agent:`
  instruction/note commits separate from source and developer-facing docs.
- Hand off the worktree path, branch, commit IDs, changed paths, exact checks and
  results, remaining limitations and any generated artifacts needed for review.
  Disclose uncommitted changes; do not describe a dirty checkout as the commit.
- Use `.agents/notes/` only for durable handoffs worth retaining beyond this task;
  otherwise use task messages and ignored temporary material.

## Integration and retirement

- One designated integrator owns the target branch. Contributors do not merge or
  cherry-pick into it concurrently. Wait for the owning agent to finish writing
  before integrating its commit; avoid copying live working files between trees.
- Review the actual diff and dependencies, then merge or cherry-pick focused
  commits. Resolve conflicts according to the intended behaviour, involving the
  relevant owner when intent is unclear. Do not overwrite unrelated local work.
- Run the checks required by the combined change on the integration checkout.
  Passing independent branch checks does not establish compatibility of the
  merged result. Report any unverified combination explicitly.
- Confirm integration, inspect tracked/untracked status and retain needed ignored
  artifacts before removing a worktree. Never use forced removal or branch deletion
  to hide incomplete work. Do not delete shared datasets or a shared `.venv` target.
