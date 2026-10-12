# Independent HTTP benchmarks

The HTTP benchmark harness, public dataset adapters and data download scripts
have moved to the private repository:

https://github.com/eveiscoding/memory-retriever-xinyi-benchmarks

Server checkout: `/srv/memory-retriever-xinyi-benchmarks`.
It imports no core-system code and tests any compatible development API using
synchronous Add/Search requests. Keep the benchmark revision, suite and input
hashes fixed across system branches. Use separate development databases; never
run routine offline benchmarks against the production endpoint.

From the independent checkout:

```bash
python3 scripts/create_run_lock.py \
  --core-repo /srv/flowgrid-aml-retriever-dev \
  --suite suites/regression.json \
  --out run_locks/core-branch.json

python3 scripts/verify_core_checkout.py \
  --core-repo /srv/flowgrid-aml-retriever-dev \
  --suite suites/regression.json \
  --lock run_locks/core-branch.json \
  --out benchmark_reports/core-branch
```

The lock refuses dirty checkouts and freezes both commits, the suite, Benchmark
code and input hashes. The verifier starts and stops its own disposable process/database
and checks the child process revision before the first Add; it does not restart
production. The default regression suite uses synthetic
data. The full public suite requires all declared input data, and missing
evidence annotations remain unscored. Final Answer-model scoring is not yet
implemented; retrieval metrics do not equal official AML scores.

Engine unit tests and implementation-specific ablations in
`aml_retriever/evaluation`, `scripts/run_eval.py` and `scripts/run_scan.py` remain
in the core repository. They inspect internals and are not the stable
cross-branch benchmark. Raw data and historical reports remain Git-ignored and
were not deleted by this source migration.
