# Live Agent Evaluation

TokenCut v0.2 adds a bounded, tool-using diagnostic agent and a paired evaluation
runner. It runs the same tasks before and after tool-output optimization and
checks structured answers against expectations that are never sent to the model.

The included tasks use synthetic, read-only fixtures. A live run uses a real model
to choose tools and answer; it is still a small demo, not evidence about your
production agent. No live results are bundled because this workspace has no
configured model API connection.

## Start Without a Key

```bash
python -m tokencut eval --dry-run
python -m tokencut eval --suite examples/evaluation_cases.json --dry-run
```

Dry runs validate the suite and show the maximum number of model calls. They do
not construct a model client, download tokenizer data, or claim measured savings.

## Run the Demo Live

From the project directory, install the optional SDK and tokenizer:

```bash
python -m pip install -e ".[tokens,live]"
python examples/run_live_eval.py
```

The launcher asks for your model name and prompts for the API key without echoing
it. The key stays in process memory and is not written to a file or report. Choose
a model available to your account that supports Chat Completions, function calls,
developer messages, and `max_completion_tokens`. API calls use your account's
billing. The default run has three cases, two variants, and a maximum of 24 calls.

If your environment already has `OPENAI_API_KEY`, use the CLI directly:

```bash
tokencut eval --model YOUR_MODEL --repeats 3 --format json -o live-evaluation.json
```

The built-in adapter uses `https://api.openai.com/v1`; it deliberately ignores
`OPENAI_BASE_URL`. It does not log credentials, retry failed API requests, execute
generated code, or access arbitrary files/tools. The only agent tools return the
fixture strings supplied by the selected suite. It does send those selected
queries, tool descriptions, and tool outputs to OpenAI during a live run.

## What Gets Compared

| Measurement | Meaning |
| --- | --- |
| Task success | Valid final JSON, matching expected fields/types, and required tool use. |
| Total tokens | Every provider-reported input and output token across all calls. |
| Tokens per successful task | Total tokens across successes and failures, divided by successes. |
| Model calls | Every attempted call, including failed attempts. |
| Elapsed time | Full agent run, including local filtering and tool execution. |
| Regressions | A baseline success that becomes an optimized failure on the same case/repeat. |

Reasoning output and cached input are reported as subsets of output/input tokens,
not added to the total. No monetary savings are inferred from token counts.
Timeouts and missing/malformed usage make totals unknown rather than zero. The
report retains known usage as a lower bound and suppresses aggregate savings.

Baseline and optimized runs start with fresh conversations. Model settings and
fixtures are identical; their order alternates across pairs. Provider caching
and model randomness can still affect results. A changed or unreported model
identity prevents a passing comparison gate.

The default gate requires every baseline and optimized run to pass its checks,
complete usage, the same reported model, disabled transport retries, and at least
30% fewer total tokens. Use `--target-savings 20` to change the threshold.
Zero savings never passes, even with a zero target. Exit status is `0` for a passed
gate or valid dry run, `1` for an unsuccessful live gate, and `2` for setup errors.

`quality_verified` remains false: passing these explicit task checks does not prove
general answer quality, production safety, or statistical non-inferiority.
Inspect `task_checks_passed`, `quality_regressions`, and the individual runs.

## Supply Your Own Tasks

Start with `examples/evaluation_cases.json`. Each case contains:

- `name`: a unique identifier.
- `query`: the actual task and requested JSON answer fields.
- `tools`: named read-only fixtures with a `description`, `content`, and `kind`.
- `expected`: independently verified answer fields, not provided to the model.
- `required_tools`: tools that must have been successfully called to pass.

Use `kind: "log"` or `"json"` for outputs you authorize TokenCut to filter. Use
`"text"` to preserve an entire tool result. All tools take an empty `{}` argument
object. Unexpected names/arguments return a tool error; they never execute code.

Top-level expected fields are required, but extra answer fields are allowed.
Nested values must match exactly, including JSON types: `false` is not `0`.
Duplicate JSON keys, Markdown wrappers, incomplete generations, refusals, and
missing required tool calls cannot pass.

```bash
tokencut eval --suite examples/evaluation_cases.json --model YOUR_MODEL \
  --repeats 3 --max-steps 4 --max-completion-tokens 1024 \
  --target-savings 30 --format json -o custom-evaluation.json
```

Reports omit answers and full messages by default. `--include-content` adds them
for debugging; these reports can contain sensitive task data. The reported request
text-token counts are local diagnostics, separate from provider usage.

## Existing Agents

For a production agent, apply `optimize_request()` immediately before each model
request, log provider usage for every call, and use the agent's real acceptance
tests. The included runner evaluates a fixture-backed diagnostic agent; it does
not attach to arbitrary frameworks or automatically execute your repository.

For another transport, implement `complete(request) -> dict` with a Chat
Completions-shaped response and no hidden retries, then pass the client to
`tokencut.evaluation.evaluate()`. Set `source = "live"` and `max_retries = 0` only
when that describes the real transport. Test doubles remain explicitly labeled
and cannot produce reported live savings.

## Implementation References

- [Function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [Chat Completions](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)
- [Token accounting](https://developers.openai.com/api/docs/guides/token-counting)
