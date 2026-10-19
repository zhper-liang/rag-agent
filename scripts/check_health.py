"""本机 live/ready 检查，不调用 LLM；非就绪时退出非零。"""
import argparse
import json
import httpx
from ragagent.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    failed = False
    with httpx.Client(base_url=args.url, timeout=cfg.api.health_timeout_s) as client:
        for kind in ('live', 'ready'):
            try:
                response = client.get(f'/api/health/{kind}')
                print(json.dumps({kind: response.json()}, ensure_ascii=False))
                failed |= response.status_code != 200
            except (httpx.HTTPError, ValueError) as exc:
                print(json.dumps({kind: {'status': 'error', 'reason': type(exc).__name__}}))
                failed = True
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
