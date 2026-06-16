"""Free, keyless on-chain data helpers for Ethereum mainnet.

Everything here talks to a **public JSON-RPC endpoint** over plain HTTP. No account,
no API key, no paid feed. We try a small list of free public nodes in turn and use the
first that answers, so the notebook keeps working even if one provider is down or rate
limits us.

What this module gives you:

* ``connect()``            - a ``web3`` client pointed at the first healthy public RPC.
* ``get_logs()``           - paginated ``eth_getLogs`` over a block range, with retry and
                             automatic range-splitting when a provider complains the window
                             is too large.
* ``decode_transfers()``   - turn raw ERC-20 ``Transfer`` event logs into a tidy DataFrame
                             (from, to, value) by decoding the ABI ourselves.
* ``load_or_fetch_transfers()`` - the cache wrapper the notebook actually calls: fetch a
                             small recent block range once, cache the decoded result to
                             ``data/`` as Parquet, and return instantly on re-runs.

An optional ``ETHERSCAN_API_KEY`` (read from the environment) is noted for fetching ABIs
or longer history, but it is **not required** anywhere in this repo. The Transfer event
signature is a fixed standard, so we decode it without anyone's help.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
from web3 import Web3

# --- Public, keyless RPC endpoints (tried in order) --------------------------
# These are free community/public nodes. We never send a key; if one is down or rate
# limits us we fall through to the next. Order is "most reliable first" in our testing.
PUBLIC_RPCS = [
    "https://ethereum-rpc.publicnode.com",
    "https://cloudflare-eth.com",
    "https://rpc.ankr.com/eth",
    "https://eth.llamarpc.com",
]

# The canonical ERC-20 Transfer event:  Transfer(address indexed from, address indexed to, uint256 value)
# Its topic0 is the keccak hash of the signature string. Every standard token emits this,
# so matching on this topic is how we find token movements without per-token ABIs.
TRANSFER_TOPIC = "0x" + Web3.keccak(text="Transfer(address,address,uint256)").hex()

# A minimal ABI: just the one event we care about. web3 uses this to decode the log.
ERC20_TRANSFER_ABI = [{
    "anonymous": False,
    "name": "Transfer",
    "type": "event",
    "inputs": [
        {"indexed": True,  "name": "from",  "type": "address"},
        {"indexed": True,  "name": "to",    "type": "address"},
        {"indexed": False, "name": "value", "type": "uint256"},
    ],
}]

# A tiny ABI for the read-only metadata calls (symbol / decimals), so we can scale raw
# integer amounts into human units. These are optional niceties, gated behind try/except.
ERC20_META_ABI = ERC20_TRANSFER_ABI + [
    {"name": "symbol",   "type": "function", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "string"}]},
    {"name": "decimals", "type": "function", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "uint8"}]},
]

_CACHE = Path(__file__).resolve().parent.parent / "data"
_CACHE.mkdir(exist_ok=True)

# Read but do not require an Etherscan key. Present only so keyed extensions are easy later.
ETHERSCAN_API_KEY = os.environ.get("ETHERSCAN_API_KEY", "")


def connect(rpcs: list[str] | None = None, timeout: int = 20) -> Web3:
    """Return a connected ``web3`` client using the first healthy public RPC.

    Tries each endpoint in turn; the first that reports a current block number wins.
    Raises ``RuntimeError`` only if every endpoint fails (e.g. no internet at all)."""
    last_err: Exception | None = None
    for url in (rpcs or PUBLIC_RPCS):
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": timeout}))
            n = w3.eth.block_number          # one real call to prove it works
            print(f"connected to {url}  (head block {n:,})")
            return w3
        except Exception as e:               # noqa: BLE001 - we genuinely want to try the next
            print(f"  {url} unavailable ({type(e).__name__}); trying next...")
            last_err = e
    raise RuntimeError(f"no public RPC responded; last error: {last_err}")


def get_logs(w3: Web3, address: str, from_block: int, to_block: int,
             topics: list | None = None, chunk: int = 500,
             max_retries: int = 4) -> list:
    """Paginated, defensive ``eth_getLogs`` over ``[from_block, to_block]``.

    Public nodes cap how many blocks (and how many results) a single ``eth_getLogs`` may
    span. We therefore walk the range in ``chunk``-block windows. If a provider rejects a
    window as too large, we halve it and retry, so the call adapts to whatever limit the
    node enforces. Returns the raw log dicts (undecoded)."""
    address = Web3.to_checksum_address(address)
    topics = topics if topics is not None else [TRANSFER_TOPIC]
    out: list = []
    start = from_block
    while start <= to_block:
        end = min(start + chunk - 1, to_block)
        size = end - start + 1
        for attempt in range(max_retries):
            try:
                logs = w3.eth.get_logs({
                    "address": address,
                    "fromBlock": hex(start),
                    "toBlock": hex(end),
                    "topics": topics,
                })
                out += list(logs)
                break
            except Exception as e:           # noqa: BLE001
                msg = str(e).lower()
                too_big = any(k in msg for k in
                              ("too large", "limit", "range", "more than", "exceed", "10000"))
                if too_big and size > 1:
                    size = max(1, size // 2)  # shrink the window and retry this start
                    end = min(start + size - 1, to_block)
                    continue
                if attempt < max_retries - 1:
                    time.sleep(0.6 * (attempt + 1))  # transient rate limit; back off
                    continue
                raise
        start = end + 1
        time.sleep(0.05)                     # be polite to the free endpoint
    return out


def decode_transfers(w3: Web3, logs: list) -> pd.DataFrame:
    """Decode raw ERC-20 ``Transfer`` logs into a tidy DataFrame.

    Each log carries the event signature in ``topics[0]``, the indexed ``from`` and ``to``
    addresses in ``topics[1:3]``, and the (non-indexed) ``value`` in ``data``. We hand the
    log to a ``web3`` contract event object, which applies the ABI and returns typed
    fields. Columns: block, tx_hash, log_index, from, to, value (raw integer)."""
    contract = w3.eth.contract(abi=ERC20_TRANSFER_ABI)
    ev = contract.events.Transfer()
    rows = []
    for lg in logs:
        try:
            d = ev.process_log(lg)
        except Exception:                    # noqa: BLE001 - skip malformed / non-standard logs
            continue
        a = d["args"]
        rows.append({
            "block": d["blockNumber"],
            "tx_hash": "0x" + d["transactionHash"].hex(),
            "log_index": d["logIndex"],
            "from": a["from"],
            "to": a["to"],
            "value": int(a["value"]),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["block", "log_index"]).reset_index(drop=True)
    return df


def token_metadata(w3: Web3, address: str) -> tuple[str, int]:
    """Best-effort ``(symbol, decimals)`` for a token. Falls back to ('TOKEN', 18)."""
    try:
        c = w3.eth.contract(address=Web3.to_checksum_address(address), abi=ERC20_META_ABI)
        return c.functions.symbol().call(), int(c.functions.decimals().call())
    except Exception:                        # noqa: BLE001
        return "TOKEN", 18


def block_times(w3: Web3, blocks: list[int]) -> pd.Series:
    """Map a set of block numbers to UTC timestamps via ``eth_getBlockByNumber``.

    One RPC call per unique block, so we pass only the blocks we actually need."""
    uniq = sorted(set(int(b) for b in blocks))
    ts = {}
    for b in uniq:
        try:
            ts[b] = w3.eth.get_block(b)["timestamp"]
        except Exception:                    # noqa: BLE001
            ts[b] = None
        time.sleep(0.02)
    return pd.Series(ts, name="timestamp")


def load_or_fetch_transfers(token: str, lookback_blocks: int = 2000,
                            chunk: int = 500, use_cache: bool = True,
                            rpcs: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """The function the notebook calls. Fetch + decode + cache ERC-20 transfers.

    Pulls the last ``lookback_blocks`` of ``Transfer`` logs for ``token``, decodes them,
    attaches block timestamps and a human-scaled ``amount`` column, and caches the result
    to ``data/`` as Parquet. On a second run it loads the cache and skips the network
    entirely, so the notebook is reproducible and fast offline.

    Returns ``(dataframe, meta)`` where ``meta`` records the symbol, decimals and the exact
    block window used (handy for honest, reproducible reporting in the notebook)."""
    token = Web3.to_checksum_address(token)
    cache = _CACHE / f"transfers_{token.lower()}_{lookback_blocks}.parquet"
    meta_cache = _CACHE / f"transfers_{token.lower()}_{lookback_blocks}.meta.json"

    if use_cache and cache.exists() and meta_cache.exists():
        df = pd.read_parquet(cache)
        meta = pd.read_json(meta_cache, typ="series").to_dict()
        print(f"loaded {len(df):,} cached transfers for {meta.get('symbol', token)} "
              f"(blocks {meta.get('from_block')}-{meta.get('to_block')})")
        return df, meta

    w3 = connect(rpcs)
    head = w3.eth.block_number
    from_block = max(0, head - lookback_blocks + 1)
    symbol, decimals = token_metadata(w3, token)
    print(f"fetching {symbol} Transfer logs over blocks {from_block:,}-{head:,} "
          f"({lookback_blocks} blocks) in {chunk}-block windows...")

    logs = get_logs(w3, token, from_block, head, chunk=chunk)
    df = decode_transfers(w3, logs)
    print(f"decoded {len(df):,} Transfer events")

    if not df.empty:
        times = block_times(w3, df["block"].tolist())
        df["timestamp"] = pd.to_datetime(df["block"].map(times), unit="s", utc=True)
        df["amount"] = df["value"] / (10 ** decimals)

    meta = {
        "token": token, "symbol": symbol, "decimals": decimals,
        "from_block": int(from_block), "to_block": int(head),
        "lookback_blocks": int(lookback_blocks), "n_transfers": int(len(df)),
    }
    df.to_parquet(cache)
    pd.Series(meta).to_json(meta_cache)
    return df, meta
