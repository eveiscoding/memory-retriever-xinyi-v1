# Frozen source-evidence 8K baseline

Profile: configs/evidence-output/source-evidence-8k-v1.json.

This is an explicit output-consumer baseline for future iteration. Ingestion,
branch13 retrieval and ranking remain unchanged. It does not enable an HTTP
Search/Answer integration or change the running service defaults.

The implementation is exactly the evaluated d060c736a2ab3e6fdfe440f454cae13a4b93b29b.
The profile freezes the successful 8192 reference-token configuration from the
fixed development comparison. The 4096 variant was not accepted. Validation
hashes identify the existing machine evidence; reports, conversations, datasets,
credentials and run locks are not distributed in this repository.

The reference budget counts rendered evidence text, provenance, role/time/span
headers and separators. Query/system/answer tokens are excluded. This is not an
answer-model billing-token budget or an answer-accuracy claim. Output selection
adds model scoring beyond the upstream Search work.

## Explicit use

Supply the existing local pair backend/tokenizer and an authenticated single-user
SearchResult. Resolve provenance under a read-only connection and release it
before scoring. Do not call this inside an open database write transaction.

~~~python
import json
from pathlib import Path
from aml_retriever.source_evidence_output import (
    resolve_sources, make_candidates, score_candidates, selected_output,
)

profile = json.loads(Path(
    'configs/evidence-output/source-evidence-8k-v1.json'
).read_text())
assert pair_backend.model_hash == profile['model']['hash']
with retriever.connection() as con:
    resolved = resolve_sources(
        con, search_result.results, user_id=user_id,
        parent_limit=profile['parent_limit'],
    )
candidates = make_candidates(
    resolved, query, pair_backend.tokenizer, **profile['windows'],
)
scored = score_candidates(candidates, query, pair_backend)
evidence = selected_output(
    resolved, scored, budget=profile['budget'],
    tokenizer=pair_backend.tokenizer, **profile['selection'],
)
~~~

Do not silently substitute a tokenizer or pair model: these affect window
boundaries, scoring and output budget. All text remains exact original character
spans; neighbor context is restricted to provenance already inside the selected
Top20. Source overlap is merged only within the same original message.

## Continue iteration

Preserve codex/20-source-evidence-8k-baseline. Record its full commit SHA and
create a new development branch from that commit for the next experiment.
Keep the evaluated implementation commit and validation hashes in the profile;
a new algorithm/profile needs a separate paired comparison. This reference is
not an official Smoke version and does not authorize deployment.
