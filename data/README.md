# Data fixtures

Cached Ethereum on-chain data: decoded ERC-20 `Transfer` logs, committed as
`transfers_<token>_<blocks>.parquet` (with a `.meta.json` sidecar) so the notebook runs
offline with no API key. The committed Parquet is the fixture itself (delete it to re-fetch); the committed example covers USDC
(`0xa0b8...eb48`).

The pipeline reads logs from a free, keyless public JSON-RPC endpoint via
`load_or_fetch_transfers()` in `src/data.py`; the cached Parquet is the first-run result.
Public on-chain data only.
