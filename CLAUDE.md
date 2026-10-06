\# SplitLab rules for Claude Code

\- core/ is pure Python: numpy, pandas, scipy, statsmodels, lightgbm.

&#x20; No FastAPI imports there, no dashboard code there.

\- Do not edit notebooks. Sabina owns the notebooks.

\- Do not change a function's numerical output without explaining why first.

\- Every function you touch needs a pytest test in tests/.

\- Never run git add or git commit. Sabina stages and commits manually.

\- For anything bigger than one function or one test file, propose a plan

&#x20; first and wait for approval before writing code.

\- SQL lives in sql/, never as inline strings inside core/.

\- Every SQL metric needs a parity test against its pandas version.

\- Never run aws commands and never read or write credentials. Sabina runs all AWS steps manually.

