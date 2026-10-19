"""从 config.yaml 生成 compose、Prometheus、Grafana 文件，不维护第二套参数。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import yaml
from ragagent.config import load_config
if __package__:
    from scripts.gen_alerts import generate_rules
else:
    from gen_alerts import generate_rules


def generate(cfg, output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    s = cfg.monitor.stack
    config_hash = hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()
    def write(name, value):
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False))
    write('alerts.yml', generate_rules(cfg.monitor.alerts))
    write('prometheus.yml', {
        'global': {'scrape_interval': f'{s.scrape_interval_s}s', 'scrape_timeout': f'{s.scrape_timeout_s}s'},
        'rule_files': ['/etc/prometheus/alerts.yml'],
        'scrape_configs': [
            {'job_name': 'ragagent', 'metrics_path': cfg.monitor.metrics.path,
             'static_configs': [{'targets': [s.api_target]}]},
            {'job_name': 'milvus', 'static_configs': [{'targets': [s.milvus_target]}]},
            {'job_name': 'gpu', 'static_configs': [{'targets': [s.gpu_target]}]},
        ],
    })
    write('provisioning/datasources/ragagent.yml', {'apiVersion': 1, 'datasources': [{
        'name': 'Prometheus', 'uid': 'ragagent-prometheus', 'type': 'prometheus', 'access': 'proxy',
        'url': f'http://127.0.0.1:{s.prometheus_port}' if s.network_mode == 'host' else f'http://prometheus:{s.prometheus_port}', 'isDefault': True, 'editable': False}]})
    write('provisioning/dashboards/ragagent.yml', {'apiVersion': 1, 'providers': [{
        'name': 'ragagent', 'type': 'file', 'options': {'path': '/var/lib/grafana/dashboards'}}]})
    dashboard = json.loads((Path(__file__).resolve().parent.parent / 'deploy/monitoring/dashboard.json').read_text())
    (out / 'dashboard.json').write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + '\n')
    compose = {'services': {
        'prometheus': {'image': s.prometheus_image, 'network_mode': 'host',
            'command': ['--config.file=/etc/prometheus/prometheus.yml',
                        f'--web.listen-address={"127.0.0.1" if s.network_mode == "host" else "0.0.0.0"}:{s.prometheus_port}',
                        '--storage.tsdb.path=/prometheus', f'--storage.tsdb.retention.time={s.retention_days}d'],
            'volumes': [f'{out / "prometheus.yml"}:/etc/prometheus/prometheus.yml:ro',
                        f'{out / "alerts.yml"}:/etc/prometheus/alerts.yml:ro', 'prometheus_data:/prometheus']},
        'grafana': {'image': s.grafana_image, 'network_mode': 'host',
            'environment': {'GF_SERVER_HTTP_ADDR': '127.0.0.1' if s.network_mode == 'host' else '0.0.0.0', 'GF_SERVER_HTTP_PORT': str(s.grafana_port),
                'GF_SECURITY_ADMIN_PASSWORD': '${' + s.grafana_password_env + ':?set Grafana password}',
                'GF_USERS_ALLOW_SIGN_UP': 'false', 'GF_AUTH_ANONYMOUS_ENABLED': 'false',
                'GF_ANALYTICS_REPORTING_ENABLED': 'false', 'GF_ANALYTICS_CHECK_FOR_UPDATES': 'false',
                'GF_PLUGINS_PREINSTALL_DISABLED': 'true'},
            'volumes': [f'{out / "provisioning"}:/etc/grafana/provisioning:ro',
                        f'{out / "dashboard.json"}:/var/lib/grafana/dashboards/ragagent.json:ro',
                        'grafana_data:/var/lib/grafana'], 'depends_on': ['prometheus']},
        }, 'volumes': {'prometheus_data': {}, 'grafana_data': {}}}
    for name, port in [('prometheus', s.prometheus_port), ('grafana', s.grafana_port)]:
        service = compose['services'][name]
        # 让 up -d 在配置改变后重建容器，避免旧抓取/规则配置继续运行。
        service['labels'] = {'ragagent.config_hash': config_hash}
        if s.network_mode == 'bridge':
            service.pop('network_mode')
            service['ports'] = [f'127.0.0.1:{port}:{port}']
            service['extra_hosts'] = ['host.docker.internal:host-gateway']
    write('compose.yml', compose)
    return out / 'compose.yml'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--output', default='deploy/monitoring/generated')
    args = parser.parse_args(argv)
    print(generate(load_config(args.config), args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
