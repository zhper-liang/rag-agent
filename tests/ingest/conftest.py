"""仅本线离线单测的配置占位key，不加载/调用真实LLM。"""
import pytest


@pytest.fixture(autouse=True)
def offline_config_keys(monkeypatch):
    for name in ('LLM_MAIN_KEY','LLM_MAIN_BACKUP_KEY','LLM_SMALL_KEY','LLM_SMALL_BACKUP_KEY'):
        monkeypatch.setenv(name,'offline-test-unused')
