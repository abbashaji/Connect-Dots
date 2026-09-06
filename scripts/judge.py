name: Judge

on:
  workflow_dispatch: {}
  # workflow_run and schedule triggers disabled while debugging the 429
  # quota issue -- don't want Judge auto-firing and burning quota too.
  # workflow_run:
  #   workflows: ["Gap Miner", "Capability Miner"]
  #   types: [completed]
  # schedule:
  #   - cron: "30 */6 * * *"

concurrency:
  group: ideation-pipeline-write
  cancel-in-progress: false

jobs:
  judge:
    runs-on: ubuntu-latest
    # Skip if triggered by a miner run that failed -- no point judging
    # against a corpus that didn't actually get a new entry this round.
    if: >
      github.event_name != 'workflow_run' ||
      github.event.workflow_run.conclusion == 'success'
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - run: pip install -r requirements.txt

      - name: Run judge
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          PYTHONUNBUFFERED: "1"
        run: python scripts/judge.py

      - name: Commit and push
        run: |
          git config user.name "judge-bot"
          git config user.email "actions@users.noreply.github.com"
          git add data/pair_verdicts.jsonl
          if git diff --cached --quiet; then
            echo "No changes to commit"
            exit 0
          fi
          git commit -m "judge: score and audit new pairs"
          git pull --rebase origin "${GITHUB_REF_NAME}"
          git push origin "HEAD:${GITHUB_REF_NAME}"
