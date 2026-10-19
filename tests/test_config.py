import copy

import pytest
import yaml

from ragagent.config import DEFAULT_PATH, ConfigError, load_config, main

RAW = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
KEYS = {ep["api_key_env"]: "sk-test" for ep in RAW["llm"]["endpoints"].values()}


def write(tmp_path, mutate=None, text=None):
    p = tmp_path / "config.yaml"
    if text is None:
        data = copy.deepcopy(RAW)
        if mutate:
            mutate(data)
        text = yaml.safe_dump(data, allow_unicode=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_real_config_loads():
    cfg = load_config(env=KEYS)
    assert cfg.llm.scenes.generate.primary in cfg.llm.endpoints
    assert cfg.index.dim == 1024
    assert cfg.monitor.alerts["ttft_p95_s"].for_ == "10m"


def test_stub_addresses_load(tmp_path):
    def to_stub(d):
        for ep in d["llm"]["endpoints"].values():
            ep["base_url"] = "http://127.0.0.1:8100/v1"
    load_config(write(tmp_path, to_stub), env=KEYS)


@pytest.mark.parametrize("field", ["base_url", "model"])
def test_empty_endpoint_field_rejected(tmp_path, field):
    p = write(tmp_path, lambda d: d["llm"]["endpoints"]["main"].__setitem__(field, ""))
    with pytest.raises(ConfigError, match="不能为空"):
        load_config(p, env=KEYS)


def test_missing_key_env_rejected():
    env = {k: v for k, v in KEYS.items() if k != "LLM_SMALL_KEY"}
    with pytest.raises(ConfigError, match="LLM_SMALL_KEY"):
        load_config(env=env)


def test_remote_http_rejected(tmp_path):
    p = write(tmp_path, lambda d: d["llm"]["endpoints"]["main"].__setitem__("base_url", "http://api.example.com/v1"))
    with pytest.raises(ConfigError, match="https"):
        load_config(p, env=KEYS)


def test_unknown_scene_endpoint_rejected(tmp_path):
    p = write(tmp_path, lambda d: d["llm"]["scenes"]["chat"].__setitem__("fallback", "nope"))
    with pytest.raises(ConfigError, match="nope"):
        load_config(p, env=KEYS)


def test_unknown_field_rejected(tmp_path):
    p = write(tmp_path, lambda d: d["rerank"].__setitem__("topn", 8))
    with pytest.raises(ConfigError, match="topn"):
        load_config(p, env=KEYS)


def test_search_list_below_top_k_rejected(tmp_path):
    p = write(tmp_path, lambda d: d["retrieval"].__setitem__("search_list", 10))
    with pytest.raises(ConfigError, match="search_list"):
        load_config(p, env=KEYS)


def test_backup_dbs_must_name_paths(tmp_path):
    p = write(tmp_path, lambda d: d["backup"].__setitem__("dbs", ["registry_db", "chunks"]))
    with pytest.raises(ConfigError, match="dbs"):
        load_config(p, env=KEYS)
    p = write(tmp_path, lambda d: d["backup"].__setitem__("dbs", ["session_db", "session_db"]))
    with pytest.raises(ConfigError, match="重复"):
        load_config(p, env=KEYS)


def test_retain_days_removed(tmp_path):
    p = write(tmp_path, lambda d: d["logging"].__setitem__("retain_days", 30))
    with pytest.raises(ConfigError, match="retain_days"):
        load_config(p, env=KEYS)


def test_duplicate_key_rejected(tmp_path):
    text = DEFAULT_PATH.read_text(encoding="utf-8") + "\ngrade:\n  pass_score: 0.9\n  max_retries: 2\n"
    with pytest.raises(ConfigError, match="重复"):
        load_config(write(tmp_path, text=text), env=KEYS)


def test_main_exit_code(tmp_path, monkeypatch, capsys):
    p = write(tmp_path, lambda d: d["llm"]["endpoints"]["main"].__setitem__("base_url", ""))
    for k, v in KEYS.items():
        monkeypatch.setenv(k, v)
    assert main([str(p)]) == 1
    assert "配置校验失败" in capsys.readouterr().err
    assert main([]) == 0


@pytest.mark.parametrize('updates', [
    {'api_target':'0.0.0.0:8000'}, {'gpu_target':'example.com:9835'},
    {'grafana_password_env':'PASSWORD:-injected'}, {'scrape_timeout_s':999},
    {'gpu_timeout_s':999}, {'grafana_port':9090}, {'network_mode':'unsafe'},
])
def test_monitoring_stack_safety(tmp_path, updates):
    p = write(tmp_path, lambda d: d['monitor']['stack'].update(updates))
    with pytest.raises(ConfigError):
        load_config(p, env=KEYS)
