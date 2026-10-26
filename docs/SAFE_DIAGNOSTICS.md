# Minimal private diagnostics

Isolated branch based on frozen Core
`9fc9c6487b36fe1586ba4a1df73eada0aa8e943e`; no indexed-write optimization
is included. Not deployed to the official Full service.

Opt in at process startup with `AML_SAFE_DIAGNOSTICS=1`. Default is off.
This requires a new process/image; it is not a live toggle for an existing
Python process. A new image must expose its actual revision, not masquerade
as the frozen image. Confirm deployment authorization and the platform's
checkpoint/version policy before replacing a registered service.

With safe diagnostics enabled, POST requests emit one JSON line to stderr
even under `--quiet`, with exactly:

- `event`: fixed `aml_timing`;
- `stage`: fixed processing-stage category (not the URL);
- `elapsed_ms`: total monotonic elapsed time for this HTTP handler;
- `error_type`: allowlisted exception class, `OtherError`, or `none`.

The stage is the failing stage, or `response` on normal completion. Thread-level
uncaught failures use `handler` with elapsed near zero (thread duration unknown),
not a fabricated request duration. There is no request/user/session ID, message,
query, URL, header, address, exception message, stack or traceback local.
Legacy handler access/trace output is suppressed in this mode even if quiet is
off. Output failures are ignored so they cannot change the API outcome.

This preserves status/response semantics and database/retrieval logic. It does
not fix slow writes, establish a timeout cause, capture a permanently hung
request before completion, or replace platform-side HTTP/error metadata.
Direct runtime termination/OOM cannot reliably generate an application event.

Before deployment: server full tests, isolated HTTP checks, and an explicitly
bounded log rotation policy (for example Docker json-file max-size 10m and
max-file 3). This logs success timings as well as errors; size retention must
be bounded. Preserve official data policy, keep logs private, and remove related
run metadata within the applicable lifecycle deadline. Do not enable broad
debug/SQL tracing, full access logs, or request-body capture.
