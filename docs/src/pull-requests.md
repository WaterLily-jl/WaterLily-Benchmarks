# Pull-request benchmarks

A maintainer of [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl) can benchmark a pull request against `master` by commenting on it:

```
/benchmark
/benchmark -c "tgv sphere" -r 3
```

optionally followed by [`benchmark.sh` options](benchmarks.md) on the same line. The workflow fetches the pull request as the branch `pr-<number>`, runs both refs on a GitHub-hosted runner, and posts the `compare.jl` table as a comment, which later runs update. By default it runs the default cases on CPUx01 (SIMD) and CPUx02 (KernelAbstractions), in 2 repetitions. A hosted runner is noisy: read `Signif` before `Δ` (see [Methodology](methodology.md)). The reference is the tip of `master`, not the merge base.

`-w` and `-bs` replace the refs, the first run being the baseline: `/benchmark -w "pr-123 pr-123" -bs "main my-branch"` compares two BiotSavartBCs.jl branches on the pull request's WaterLily (see [BiotSavartBCs changes](biotsavart.md)). The workflow can also be started from the Actions tab of WaterLily.jl (`workflow_dispatch`), with the pull request number and the arguments.

## The action

[`action.yml`](https://github.com/WaterLily-jl/WaterLily-Benchmarks/blob/main/action.yml) makes this repository a composite GitHub Action, `uses: WaterLily-jl/WaterLily-Benchmarks@main`. Given a WaterLily.jl checkout with two refs, it checks out this repository next to it (the checkpoints are fetched on demand), installs Julia, runs `benchmark.sh -w "<base> <head>" -u true` with the given arguments and `compare.jl --markdown --speedup_base=<base>`, so that `Δ ± σ` and `Signif` compare `head` with `base` on each backend. The report goes to the job summary, to an artifact with the JSON results and, given a pull request number, to a comment that later runs update (one per `label`).

| Input | Default | Meaning |
|---|---|---|
| `waterlily-dir` | the workspace | WaterLily.jl checkout containing `base` and `head` |
| `base`, `head` | `master`, (required) | refs to compare, e.g. `head: pr-123` |
| `args` | none | `benchmark.sh` arguments appended to the defaults `-b Array -t "1 2" -r 2`, a later value winning, e.g. `-c sphere -r 3`; restricted to `[A-Za-z0-9 ,._="-]`. With `-bs`, BiotSavartBCs.jl is checked out next to WaterLily, so its refs must exist on GitHub |
| `force` | `true` | pass `-f` (skip the machine check), for shared runners |
| `julia-version` | `1.11` | for `julia-actions/setup-julia` |
| `harness-ref`, `harness-path` | the action's ref, `WaterLily-Benchmarks` | revision of this repository to run, and its checkout path |
| `pr`, `label` | none | pull request to comment on, and the name of the comment |
| `artifact` | `benchmark-results` | artifact with the JSON results and the report, uploaded also when a step fails; empty uploads nothing |
| `token` | `github.token` | needs `pull-requests: write`; only the commenting step sees it |

The outputs are `report`, the path of the markdown report, and `data`, the directory of the JSON results.

## Calling it

WaterLily.jl's [`benchmark.yml`](https://github.com/WaterLily-jl/WaterLily.jl/blob/master/.github/workflows/benchmark.yml) is the full caller. The minimal one is:

```yaml
on:
  issue_comment: {types: [created]}
permissions:
  contents: read
  pull-requests: write
jobs:
  benchmark:
    if: github.event.issue.pull_request && startsWith(github.event.comment.body, '/benchmark') &&
        contains(fromJSON('["OWNER", "MEMBER", "COLLABORATOR"]'), github.event.comment.author_association)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
        with: {ref: master, persist-credentials: false}
      - run: git fetch origin "pull/${{ github.event.issue.number }}/head:pr-${{ github.event.issue.number }}"
      - uses: WaterLily-jl/WaterLily-Benchmarks@main
        with:
          head: pr-${{ github.event.issue.number }}
          pr: ${{ github.event.issue.number }}
```

## Security

The pull request's code runs on the runner, so only maintainers who have read the diff can trigger it (owner, member or collaborator), the workflow file always comes from `master`, the benchmark step sees no token, and both checkouts use `persist-credentials: false`. The arguments are checked against the same character set in the workflow and in the action before they reach the shell. On a self-hosted machine, add a `concurrency` group, use an unprivileged ephemeral runner with no secrets, and set `force: false` so that the machine check runs.

## Testing changes to the action

[`action-test.yml`](https://github.com/WaterLily-jl/WaterLily-Benchmarks/blob/main/.github/workflows/action-test.yml) runs the action of a branch of this repository against WaterLily.jl, `master` against `master~N`, with a short sweep and no comment. Start it from the Actions tab, or with `gh workflow run action-test.yml --ref <branch>`, before WaterLily.jl relies on a change.
