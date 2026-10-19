"""nvidia-smi 本机 exporter；无 DCGM/容器 GPU 运行时依赖。"""
import csv
import io
import subprocess

from prometheus_client import CollectorRegistry, Gauge, CONTENT_TYPE_LATEST, generate_latest
from fastapi import FastAPI, Response


def gpu_metrics(timeout):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True,
                            check=True, timeout=timeout)
    registry = CollectorRegistry()
    used = Gauge('rag_gpu_memory_used_bytes', 'GPU used framebuffer bytes', ['gpu'], registry=registry)
    total = Gauge('rag_gpu_memory_total_bytes', 'GPU total framebuffer bytes', ['gpu'], registry=registry)
    count = 0
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row:
            continue
        index, used_mib, total_mib = (part.strip() for part in row)
        if not index.isdecimal():
            raise ValueError('invalid gpu index')
        u, t = float(used_mib), float(total_mib)
        if not 0 <= u <= t or t <= 0:
            raise ValueError('invalid GPU memory values')
        count += 1
        used.labels(gpu=index).set(u * 1024 ** 2)
        total.labels(gpu=index).set(t * 1024 ** 2)
    if not count:
        raise ValueError('no GPU metrics')
    return generate_latest(registry)


def create_app(cfg=None):
    from ragagent.config import load_config
    cfg = cfg or load_config()
    app = FastAPI()

    @app.get('/metrics', include_in_schema=False)
    def metrics():
        try:
            return Response(gpu_metrics(cfg.monitor.stack.gpu_timeout_s),
                            headers={'Content-Type': CONTENT_TYPE_LATEST})
        except (OSError, subprocess.SubprocessError, ValueError):
            return Response('GPU metrics unavailable', status_code=503)
    return app


def main():
    import uvicorn
    from ragagent.config import load_config
    cfg = load_config()
    uvicorn.run(create_app(cfg), host='127.0.0.1', port=cfg.monitor.stack.gpu_port)


if __name__ == '__main__':
    main()
