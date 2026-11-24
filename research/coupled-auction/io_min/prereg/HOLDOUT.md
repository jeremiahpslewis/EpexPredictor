# Held-out window for the out-of-sample accuracy test (fixed 2026-11-24, before any holdout record is built)

- **Holdout:** MTUs from 2026-09-08 00:00 UTC to 2026-11-08 23:45 UTC, the last two months of the sample.
- **Training:** 2025-10-01 to 2026-09-07. Every fit, specification choice and tuning step uses training data only.
- **Use of the holdout.** It is scored once per frozen specification. Its outcome fields (prices, net positions,
  schedules, shadow prices, flows) are not read until the specification and the scoring rules are committed.
- **Already seen.** The existing eleven-zone fit, the outage study and the slope check used MTUs from the whole period,
  including this window. The window is therefore clean only for refits on the training data, not for the current
  posterior.
