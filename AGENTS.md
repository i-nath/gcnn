# AGENTS.md

This file tells Codex-style coding agents how to work in this repository.

## Purpose

The goal in this repository is to make safe, useful progress with minimal back-and-forth:

- understand the code before changing it
- prefer small, correct, testable edits
- preserve existing patterns unless there is a strong reason to improve them
- leave the repo easier for the next person to work in

## Working Agreement

- Be proactive. If the request is implementation-oriented, make the change instead of only describing it.
- Be conservative with risk. Avoid broad refactors unless the task clearly calls for them.
- Do not overwrite or revert user changes you did not make.
- When requirements are ambiguous, choose the smallest reasonable interpretation and state assumptions briefly in the final message.
- If you discover a real tradeoff or hidden risk, pause and surface it clearly before taking the risky path.

## First Steps

Before editing:

1. Identify the relevant files and read surrounding code.
2. Look for existing conventions in naming, structure, tests, and error handling.
3. Check whether there are nearby docs, configs, or examples that define the intended pattern.
4. Prefer local fixes over speculative architecture changes.

## Editing Rules

- Keep diffs focused and easy to review.
- Match the existing style of the file and project.
- Avoid renaming or moving files unless it materially helps the task.
- Do not add headers, banners, license blocks, or boilerplate unless requested.
- Prefer straightforward code over clever abstractions.
- Add comments only when they clarify non-obvious intent.
- Avoid introducing new dependencies unless they are clearly justified.

## Testing And Verification

After making changes:

1. Run the narrowest relevant checks first.
2. If there are targeted tests for the changed area, run those before broader suites.
3. If no tests exist, verify behavior with the most direct available command or reasoning.
4. Report what you verified and what you could not verify.

## Git Safety

- Never use destructive git commands such as `git reset --hard` or `git checkout --` unless explicitly requested.
- Do not amend commits unless explicitly requested.
- Ignore unrelated dirty files unless they directly block the task.
- If you create a branch, use the `codex/` prefix.

## Communication

While working:

- Send short progress updates during longer tasks.
- Explain what you are about to change before editing files.
- Keep explanations concise and practical.

In the final response:

- summarize the user-visible outcome
- mention key files changed
- note verification performed
- call out any remaining risks or follow-ups only if they matter

## Code Review Mode

If the user asks for a review:

- prioritize bugs, regressions, missing validation, and test gaps
- list findings first, ordered by severity
- include precise file references when possible
- keep summaries brief

## Preferred Approach

- Search with `rg` / `rg --files` when available.
- Reuse existing utilities and patterns before creating new ones.
- Prefer targeted tests over exhaustive runs during iteration.
- For frontend work, preserve the app's visual language unless the task is explicitly a redesign.

## When To Ask The User

Ask only when a decision would likely:

- change public behavior in a non-obvious way
- require a destructive action
- introduce a new dependency or significant refactor
- depend on product intent that cannot be inferred from the codebase

Otherwise, make a reasonable assumption and keep moving.

## Definition Of Done

A task is done when:

- the requested change is implemented
- obvious affected code paths are updated
- relevant checks have been run when possible
- the final message clearly states what changed and any important caveats

