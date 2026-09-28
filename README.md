\# SplitLab

An A/B testing and causal inference platform for product decisions.



\## Scenario

A short-video feed is testing a new ranking model against the current one.



\## Hypothesis

The new ranker increases watch time per user by at least 2% with no harm to guardrails.



\## Metrics

\- \*\*Primary:\*\* watch time per user

\- \*\*Guardrails:\*\* next-day retention, creator diversity, latency

\- \*\*Secondary:\*\* likes, shares



\## Success criteria

Ship if the primary metric improves (CI excludes 0 at 95%, corrected for peeking),

no guardrail is breached, and the SRM check passes. Otherwise hold or iterate.



\## Data

KuaiRand-Pure (real feed logs), Open Bandit Dataset, and a seeded simulator.

The experiment is simulated; real data validates the methods.

