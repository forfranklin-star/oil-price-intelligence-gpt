from __future__ import annotations
import json
from src.providers.diesel import ChinaDieselProvider
from src.utils import load_config

if __name__ == "__main__":
    p = ChinaDieselProvider(load_config())
    results = p.collect_all()
    print(json.dumps(p.diagnostics, ensure_ascii=False, indent=2, default=str))
    if results:
        print("\nAvailable real diesel sources:")
        for r in results:
            latest = r.frame.sort_values("date").iloc[-1]
            print(f"- {r.name:18s} {r.server:24s} {r.metric:30s} rows={len(r.frame):4d} latest={latest['date'].date()} price={latest['price']}")
    else:
        raise SystemExit("No source returned a fresh real observation")
