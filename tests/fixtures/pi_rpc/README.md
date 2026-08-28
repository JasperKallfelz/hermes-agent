# Pi RPC 0.84.3 fixtures

Sanitized protocol records captured from the pinned RPC shapes and used by
unit tests. Dynamic IDs, paths, provider/model names, usage, and content are
replaced with fixture values. They contain no account, token, credential, or
user-path data. A live model prompt remains gated on separate Pi authentication.

Supported protocol matrix:

| Pi package | Status | Required fences/surface |
| --- | --- | --- |
| `@earendil-works/pi-coding-agent@0.84.3` | supported, exact pin | literal-LF JSONL, correlated `response.command`, persistent `sessionId`/`sessionFile`, `get_entries.leafId`, `agent_start` + `agent_settled`, extension UI protocol |
| Any other version | rejected | re-run protocol fixtures, doctor, authenticated E2E, and security review before changing the pin |
