import json
from pathlib import Path
import yaml
from scripts.gen_monitoring import generate
from tests.integration.helpers import config


def test_generated_stack_uses_config_only_and_loopback(tmp_path):
    cfg = config(tmp_path)
    cfg = cfg.model_copy(update={'monitor':cfg.monitor.model_copy(update={'stack':cfg.monitor.stack.model_copy(
        update={'prometheus_port':19090, 'grafana_port':13000, 'gpu_port':19835, 'retention_days':9, 'gpu_target':'host.docker.internal:19835'})})})
    path = generate(cfg, tmp_path/'generated')
    compose = yaml.safe_load(path.read_text())
    prom, graf = compose['services']['prometheus'], compose['services']['grafana']
    assert '--web.listen-address=0.0.0.0:19090' in prom['command']
    assert prom['ports'] == ['127.0.0.1:19090:19090']
    assert graf['ports'] == ['127.0.0.1:13000:13000']
    assert '--storage.tsdb.retention.time=9d' in prom['command']
    assert graf['environment']['GF_SERVER_HTTP_ADDR'] == '0.0.0.0'
    assert graf['environment']['GF_SERVER_HTTP_PORT'] == '13000'
    assert len(prom['labels']['ragagent.config_hash']) == 64
    assert graf['environment']['GF_SECURITY_ADMIN_PASSWORD'].startswith('${GRAFANA_ADMIN_PASSWORD:')
    scrape = yaml.safe_load((path.parent/'prometheus.yml').read_text())
    assert len(scrape['scrape_configs']) == 3
    assert scrape['scrape_configs'][2]['static_configs'][0]['targets'] == ['host.docker.internal:19835']
    assert '/etc/prometheus/alerts.yml' in scrape['rule_files']
    dashboard = json.loads((path.parent/'dashboard.json').read_text())
    assert len(dashboard['panels']) == 5
    assert len(dashboard['panels'][0]['targets']) == 3
    provision = yaml.safe_load((path.parent/'provisioning/datasources/ragagent.yml').read_text())
    assert provision['datasources'][0]['url'] == 'http://prometheus:19090'


def test_native_linux_host_mode_binds_loopback(tmp_path):
    cfg = config(tmp_path)
    cfg = cfg.model_copy(update={'monitor':cfg.monitor.model_copy(update={'stack':cfg.monitor.stack.model_copy(
        update={'network_mode':'host', 'api_target':'127.0.0.1:8000', 'milvus_target':'127.0.0.1:9091',
                'gpu_target':'127.0.0.1:9835'})})})
    compose = yaml.safe_load(generate(cfg, tmp_path/'generated').read_text())
    prom, graf = compose['services']['prometheus'], compose['services']['grafana']
    assert prom['network_mode'] == 'host'
    assert '--web.listen-address=127.0.0.1:9090' in prom['command']
    assert graf['environment']['GF_SERVER_HTTP_ADDR'] == '127.0.0.1'
    assert 'ports' not in prom
