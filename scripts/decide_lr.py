"""Pre-registered LR decision for the 4B program (conf/experiment/p4b_lr.yaml).

Rule (fixed before the sweep ran): the LR whose random arm has the highest
mean held-out (otval200) strict avg@4 over steps 15 and 30; near-ties
(< 1 point below the best) go to the LOWER LR. Never reads MATH-500.

  python scripts/decide_lr.py outputs/runs/p4b-lr   -> writes <run>/chosen_lr.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf


def main(run: Path) -> None:
    cfg = OmegaConf.load(run / "resolved_config.yaml")
    lrs = {str(a): float(s.learning_rate) for a, s in cfg.driver.arms.items()}
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    acc = defaultdict(dict)
    for r in rows:
        acc[r["arm"]][r["step"]] = r["eval"]["otval200"]["strict_avg_at_n"]
    table = {}
    for arm, by_step in acc.items():
        if not {15, 30} <= set(by_step):
            raise SystemExit(f"{arm}: missing step 15/30 evals ({sorted(by_step)})")
        table[arm] = {"lr": lrs[arm], "step0": by_step.get(0), "step15": by_step[15], "step30": by_step[30],
                      "mean_15_30": (by_step[15] + by_step[30]) / 2}
    best = max(v["mean_15_30"] for v in table.values())
    eligible = [v for v in table.values() if v["mean_15_30"] >= best - 0.01]
    chosen = min(eligible, key=lambda v: v["lr"])["lr"]
    out = {"rule": "max mean otval200 strict avg@4 over steps 15,30; ties within 1 pt -> lower LR",
           "table": table, "chosen_lr": chosen}
    (run / "chosen_lr.json").write_text(json.dumps(out, indent=2))
    for arm, v in sorted(table.items(), key=lambda kv: kv[1]["lr"]):
        print(f"{arm:10s} lr {v['lr']:.0e}  step0 {v['step0']}  s15 {v['step15']:.4f}  s30 {v['step30']:.4f}  mean {v['mean_15_30']:.4f}")
    print(f"chosen_lr {chosen}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
