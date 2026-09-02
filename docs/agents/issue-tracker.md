# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues on **lynxist-mkp/wenmai-assistant**. Use the `gh` CLI for all operations, always with `--repo lynxist-mkp/wenmai-assistant` (the Cursor workspace root is not a git clone; the clone is `wenmai-assistant/`).

## Conventions

- **Create an issue**: `gh issue create --repo lynxist-mkp/wenmai-assistant --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --repo lynxist-mkp/wenmai-assistant --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --repo lynxist-mkp/wenmai-assistant --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --repo lynxist-mkp/wenmai-assistant --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --repo lynxist-mkp/wenmai-assistant --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --repo lynxist-mkp/wenmai-assistant --comment "..."`

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

GitHub shares one number space across issues and PRs, so a bare `#42` may be either — resolve with `gh pr view 42 --repo lynxist-mkp/wenmai-assistant` and fall back to `gh issue view 42 --repo lynxist-mkp/wenmai-assistant`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue on lynxist-mkp/wenmai-assistant.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --repo lynxist-mkp/wenmai-assistant --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body. `gh issue create --repo lynxist-mkp/wenmai-assistant --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue (`gh api` on the sub-issues endpoint). Where sub-issues aren't enabled, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: GitHub's **native issue dependencies** — the canonical, UI-visible representation. Add an edge with `gh api --method POST repos/lynxist-mkp/wenmai-assistant/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric **database id** (`gh api repos/lynxist-mkp/wenmai-assistant/issues/<n> --jq .id`, _not_ the `#number` or `node_id`). GitHub reports `issue_dependencies_summary.blocked_by` (open blockers only — the live gate). Where dependencies aren't available, fall back to a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children (`gh issue list --repo lynxist-mkp/wenmai-assistant --state open`, scoped to the map's sub-issues / task list), drop any with an open blocker (`issue_dependencies_summary.blocked_by > 0`, or an open issue in the `Blocked by` line) or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --repo lynxist-mkp/wenmai-assistant --add-assignee @me` — the session's first write.
- **Resolve**: `gh issue comment <n> --repo lynxist-mkp/wenmai-assistant --body "<answer>"`, then `gh issue close <n> --repo lynxist-mkp/wenmai-assistant`, then append a context pointer (gist + link) to the map's Decisions-so-far.
