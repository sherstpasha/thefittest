# Collaborative Development in thefittest

## Iteration Scope

- Keep changes focused and minimal. Each iteration must be small enough for the user to personally read and review all changed code.
- Break substantial features into stages: minimal structure and contracts, implementation and integration of the main components, then refinements. Do not create empty layers merely to follow this sequence.
- At the start of the task, briefly state the next stage and its expected outcome.
- If a task requires multiple stages, complete only the first small, self-contained stage and stop for the user's review. If the task fits within a single iteration, complete it fully.
- If the user explicitly defines the scope of the current iteration, follow it.
- Do not include unrelated refactoring, extra features, or speculative preparation.
- At the end, briefly explain what changed, how it was verified, and the proposed next step.

## Code and Architecture

- Before making changes, study the relevant existing code, its conventions, dependencies, and architectural patterns.
- Follow the Zen of Python: explicit is better than implicit, simple is better than complex, and flat is better than nested. Prefer transparent logic even at the cost of minor code duplication, provided computational efficiency is preserved.
- Use classes for state and process orchestration, and functions for individual operations. Do not introduce a class where a function is sufficient.
- The main method should read as a sequence of clear domain steps. Keep the order of computations and state changes explicit; use named intermediate values.
- Introduce inheritance for actual shared behavior across multiple implementations, overriding only the steps that differ. Prefer simple functions and dictionaries for selecting individual operations when they are sufficient.
- Keep the public interface compact; prefix internal steps with `_`.
- Accept parameters explicitly in constructors, store configuration, and prepare the necessary state. Run the main process through a separate method. Judge constructor complexity by its responsibilities and work performed, rather than parameter count alone; do not hide parameters in a generic configuration object or `**kwargs` merely to shorten the signature.
- Keep the structure moderately modular: separate distinct responsibilities without splitting code into many small files with no practical benefit.
- Account for known growth directions when choosing component boundaries and interfaces to avoid major rework. Do not implement future features or general-purpose mechanisms without a current need.
- Do not introduce a new architectural approach without a practical reason. If existing code conflicts with these guidelines, do not automatically rewrite it.
- If an architectural decision has substantially different options with important tradeoffs, briefly explain them and ask the user to choose before implementation. Make small local decisions that do not change public contracts independently.
- Add type hints to function and method parameters and return values. Annotate variables where doing so clarifies their types; avoid unnecessary `Any`.
- Format Python code with Black, respecting the project's configuration.

## Documentation and Comments

- Create README files, guides, and other standalone instructions only when explicitly requested by the user.
- Do not leave temporary comments, documentation placeholders, `TODO` markers, or explanations of the obvious. Omit unnecessary explanations; keep necessary ones concise and relevant.
- Comments and docstrings must describe lasting purpose, contracts, or non-obvious implementation decisions. Do not record iteration progress, unfinished work, or future implementation steps in source documentation (for example, "currently only initialization is implemented" or "still needs implementation"). Report such status in the conversation instead. Do not describe planned behavior as already implemented.
- Class, function, and method docstrings are optional. Add them when they explain purpose, behavior, limitations, or non-obvious parameters.
- Do not create docstrings that merely restate a function's name, signature, or obvious actions.
- When a docstring is necessary, use NumPy-style (numpydoc) exclusively in English.
- Include `Parameters`, `Attributes`, `Returns`, `Raises`, and other sections only when needed.
- When changing code behavior, update the related docstrings without rewriting unrelated documentation.

## Verification

- Add tests in proportion to the risk and complexity of the change. For small changes, limit testing to key scenarios and minimal smoke tests.
- When fixing a bug, add a regression test reproducing the original problem where feasible.
- Do not inflate test infrastructure, duplicate implementation logic in tests, or expand coverage of unrelated components.
- Leave detailed testing, coverage expansion, and extensive edge-case matrices for a separate stage requested by the user, unless they are necessary to safely verify the current change.
- Run existing checks relevant to the change and necessary formatting and type checks if configured.
- Report only checks actually performed, their results, and significant limitations.
