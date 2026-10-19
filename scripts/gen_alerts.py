"""monitor.alerts → Prometheus rule YAML；数值、窗口、持续时间只读配置。"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import yaml

from ragagent.config import load_config

NAMES = {"ttft_p95_s", "error_ratio", "llm_fallback_ratio", "degraded_ratio", "milvus_rss_gb", "gpu_mem_ratio", "downvote_ratio"}


def generate_rules(alerts) -> dict:
    if set(alerts) != NAMES:
        raise ValueError("monitor.alerts must contain exactly the documented alert keys")
    rules = []
    for key, raw in alerts.items():
        a = raw.model_dump(by_alias=True) if hasattr(raw, "model_dump") else raw
        value = float(a["value"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("alert value must be finite and nonnegative")
        window = a.get("window") or a.get("for")
        if not window or not re.fullmatch(r"(?:[0-9]+(?:ms|s|m|h|d|w|y))+", window):
            raise ValueError("invalid alert duration")
        if "ratio" in key and value > 1:
            raise ValueError("ratio must not exceed one")
        reqs = f"clamp_min(sum(rate(rag_requests_total[{window}])), 1e-12)"
        calls = f"clamp_min(sum(rate(rag_llm_calls_total[{window}])), 1e-12)"
        expressions = {
            "ttft_p95_s": f'histogram_quantile(0.95, sum by (le) (rate(rag_stage_seconds_bucket{{stage="ttft"}}[{window}]))) > {value}',
            "error_ratio": f'sum(rate(rag_requests_total{{status="error"}}[{window}])) / {reqs} > {value}',
            "llm_fallback_ratio": f'sum(rate(rag_llm_calls_total{{outcome="fallback"}}[{window}])) / {calls} > {value}',
            "degraded_ratio": f'sum by (kind) (rate(rag_degraded_total[{window}])) / scalar({reqs}) > {value}',
            "milvus_rss_gb": f'process_resident_memory_bytes{{job="milvus"}} > {value * 1024 ** 3}',
            "gpu_mem_ratio": f'rag_gpu_memory_used_bytes / clamp_min(rag_gpu_memory_total_bytes, 1e-12) > {value}',
        }
        if key == "downvote_ratio":
            minimum = a.get("min_count")
            if not isinstance(minimum, int) or minimum < 1:
                raise ValueError("downvote_ratio requires positive min_count")
            total = f"sum(increase(rag_feedback_total[{window}]))"
            expr = f'sum(increase(rag_feedback_total{{rating="-1"}}[{window}])) / clamp_min({total}, 1e-12) > {value} and {total} >= {minimum}'
        else:
            expr = expressions[key]
        rule = {"alert": "Rag" + "".join(s.capitalize() for s in key.split("_")), "expr": expr,
                "labels": {"severity": "warning"}, "annotations": {"summary": f"{key} exceeds monitor.alerts threshold"}}
        if a.get("for"):
            if not re.fullmatch(r"(?:[0-9]+(?:ms|s|m|h|d|w|y))+", a["for"]):
                raise ValueError("invalid alert for")
            rule["for"] = a["for"]
        rules.append(rule)
    return {"groups": [{"name": "ragagent", "rules": rules}]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(generate_rules(cfg.monitor.alerts), allow_unicode=True, sort_keys=False))
    print(f"generated {len(NAMES)} alert rules: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
