# Issue tracker

Use GitHub issues on `lynxist-mkp/wenmai-assistant`.

## Default rules

1. Use `gh` for every issue or PR operation.
2. Pass `--repo lynxist-mkp/wenmai-assistant` on every command.
3. The Cursor workspace root is not the git clone; repo work happens in `lab-knowledge-assistant/`.
4. Treat a bare `#42` as ambiguous until you check whether it is a PR or an issue.

Completion check: the command you are about to run names the repo explicitly and targets the right object type.

## Core operations

- Create issue: `gh issue create --repo lynxist-mkp/wenmai-assistant --title "..." --body "..."`
- Read issue with comments: `gh issue view <number> --repo lynxist-mkp/wenmai-assistant --comments`
- List issues: `gh issue list --repo lynxist-mkp/wenmai-assistant ...`
- Comment: `gh issue comment <number> --repo lynxist-mkp/wenmai-assistant --body "..."`
- Add or remove labels: `gh issue edit <number> --repo lynxist-mkp/wenmai-assistant --add-label "..."` / `--remove-label "..."`
- Close: `gh issue close <number> --repo lynxist-mkp/wenmai-assistant --comment "..."`

For multi-line bodies, use a heredoc instead of escaping line breaks inline.

## PR or issue

This repo does not treat PRs as a request surface for triage.

When a reference like `#42` appears, resolve it in this order:

1. `gh pr view 42 --repo lynxist-mkp/wenmai-assistant`
2. If that fails, `gh issue view 42 --repo lynxist-mkp/wenmai-assistant`

## Skill hooks

- "Publish to the issue tracker" means create a GitHub issue in `lynxist-mkp/wenmai-assistant`.
- "Fetch the relevant ticket" means `gh issue view <number> --repo lynxist-mkp/wenmai-assistant --comments`.

## Wayfinding

Used by `/wayfinder`. The map is one issue; child tickets hang off that map.

1. Map: create or find one issue labeled `wayfinder:map`. It holds Notes, Decisions-so-far, and Fog.
2. Child ticket: create a normal issue, then link it as a GitHub sub-issue. If sub-issues are unavailable, add it to the map task list and put `Part of #<map>` at the top of the child body.
3. Child label: use `wayfinder:<type>` where type is `research`, `prototype`, `grilling`, or `task`.
4. Blocking: prefer native issue dependencies. Use the blocker's database id from `gh api repos/lynxist-mkp/wenmai-assistant/issues/<n> --jq .id`, then post the dependency edge. If dependencies are unavailable, fall back to a `Blocked by: #<n>` line in the child body.
5. Frontier query: among the map's open children, skip anything assigned or still blocked; first remaining item in map order wins.
6. Claim: `gh issue edit <n> --repo lynxist-mkp/wenmai-assistant --add-assignee @me`
7. Resolve: comment with the answer, close the issue, then append a context pointer to the map's Decisions-so-far.

Completion check: a wayfinding ticket is only done once the child issue is closed and the map records the result.
