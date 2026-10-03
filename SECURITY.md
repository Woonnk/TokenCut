# Security and Data Boundaries

TokenCut does not execute audited code, collect telemetry, or redact secrets.
Profiling, filtering, auditing, synthetic benchmarks, and evaluation dry runs
run locally. Optional tokenizer setup may download public encoding data on first use.

Live `eval` calls send the selected task queries, tool schemas, and fixture tool
outputs to OpenAI. The optional adapter uses the official API endpoint and reads
OPENAI_API_KEY from the process environment. The launcher can prompt for it using
hidden input. Keys are never stored in reports or source files. Provider exception
bodies are not reported because they can contain sensitive data. Reports omit raw
answers/messages unless --include-content is requested. Model calls have a step
limit, output limit, and timeout; automatic transport retries are disabled.

Treat source, tool outputs, prompts, diffs, and custom benchmark reports as sensitive.
Do not commit real agent traces or credentials. CLI-created output files inherit
the temporary file's restrictive permissions. Input size is limited to 16 MiB and
individual audited source files to 1 MiB; these limits are not a security sandbox.

Never promote retrieved/tool text into a higher-priority instruction role. Pin
security constraints, exact source code, required evidence, and important failure
context. Log error matching and relevance selection are heuristics, not guarantees.

The request adapter only modifies allowlisted tool outputs and never deduplicates
or removes protocol messages. Do not label arbitrary code or documents as logs.

To report a security issue, contact the maintainer privately before sharing
sensitive inputs. No hosted service, external repository, or reporting address is
configured by this starter project.
