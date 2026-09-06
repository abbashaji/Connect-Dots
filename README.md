# Combinatorial Innovation Pipeline (GitHub-native)

Three scheduled workflows, one shared git repo as the database. No hosted
database, no MCP layer, no manual context-switching — this replaces the
earlier fresh-chat/file-relay design entirely, since this repo *is* a
persistent execution substrate.

## Layout

```
.github/workflows/
  gap-miner.yml          -- scheduled, sources one new gap
  capability-miner.yml   -- scheduled, sources one new capability
  judge.yml              -- triggered by either miner finishing, or its own schedule
scripts/
  common.py              -- shared helpers (JSONL I/O, Gemini calls, cosine similarity)
  gap_miner.py           -- isolated: never reads capabilities.jsonl
  capability_miner.py    -- isolated: never reads gaps.jsonl
  judge.py               -- retrieval -> blind score -> gate -> elaborate -> audit
requirements.txt
data/                    -- created automatically on first run
  gaps.jsonl
  capabilities.jsonl
  pair_verdicts.jsonl
```

## One-time setup

1. Commit this structure to your repo (adjust paths if you already have a
   `scripts/` directory in use for something else).
2. Repo Settings -> Secrets and variables -> Actions -> add `GEMINI_API_KEY`.
3. That's it. `workflow_dispatch` is enabled on all three, so you can
   trigger a full manual run from the Actions tab immediately rather than
   waiting for the first scheduled tick.

## Why the concurrency group matters

All three workflows share `concurrency: group: ideation-pipeline-write`.
This is not per-workflow — it's pipeline-wide. Only one of Gap Miner,
Capability Miner, or Judge can be mid-run at any moment, repo-wide. That's
what turns "two workflows both try to `git push` at once" from a race into
a queue: the second one waits for the first to finish, rebases cleanly,
and pushes. `cancel-in-progress: false` means a queued run waits rather
than getting cancelled.

## The pipeline, end to end

1. **Gap Miner** and **Capability Miner** run independently (same cron
   cadence, but the concurrency group means they never literally overlap
   mid-write). Each is a single grounded-search call to Gemini, writing
   one new structured entry + embedding to its own `.jsonl` file. Neither
   script imports or reads the other's data file — that isolation is
   enforced by what the code does, not by an instruction either script
   could ignore.
2. **Judge** runs automatically once either miner's workflow completes
   successfully (`workflow_run` trigger), plus a schedule as backstop in
   case a run's `workflow_run` trigger is ever missed. It:
   - shortlists only the top-`TOP_K` unseen capability matches per gap by
     embedding cosine similarity (this is what avoids re-scanning every
     gap against every capability as both corpora grow)
   - skips any `(gap_id, capability_id)` pair already present in
     `pair_verdicts.jsonl` — permanent, file-based dedup
   - scores fit *blind* — the scoring call has no idea an elaboration
     step exists, so there's no reward gradient pushing scores toward
     "yes"
   - only pairs scoring >= `THRESHOLD` (70, in `judge.py`) proceed to the
     four-stage elaboration, which pulls in similar past verdicts as
     precedent context first — this is the "gains wisdom" mechanism, and
     it's retrieval-augmented, not model training; no weights change,
     ever
   - runs an entailment check (does the claimed bridge trace back to
     stated operations, or did something leak in from elsewhere), a
     search-verified existence check (stops here if the bridge already
     exists — no point auditing something already built), a perturbation
     check (does the answer move when a premise is altered), and an
     adversarial cross-exam with zero investment in the idea

## Tunable knobs (all in `scripts/judge.py`)

- `TOP_K` — how many capability candidates get shortlisted per gap.
  Higher = more thorough, more API calls, slower.
- `THRESHOLD` — the blind-verdict cutoff for proceeding to elaboration.
- Cron cadence — edit the `cron:` lines in the three workflow files.
  `0 */6 * * *` = every 6 hours; adjust to your API budget.

## Known limitations, stated plainly

- **Isolation between the two miners is real but not absolute.** Both
  draw on the same underlying model and training data. If a gap and
  capability are strongly associated in the world already, independent
  searches can still converge on both halves of a known pairing without
  ever sharing context. The existence check exists specifically to catch
  this after the fact — it is not a leakage *filter* so much as a leakage
  *filter of last resort*, and should be read that way.
- **Concurrent workflow runs still cost API calls even when queued.** The
  concurrency group prevents git-push races, not wasted spend — if all
  three trigger in the same window, they'll simply run one after another
  rather than in parallel, taking longer but not costing less.
- **`git pull --rebase` before push assumes a linear history on this
  branch.** If you also have humans committing directly to the same
  branch this pipeline writes to, a rebase conflict is possible (rare,
  since these bots only ever append to their own specific files, but not
  impossible if a human edits the same `.jsonl` file by hand).
