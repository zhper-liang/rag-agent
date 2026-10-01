"""本线桩配置；不接触真实密钥、资料或接入点。"""
from pathlib import Path

import yaml

from ragagent.config import Config

ROOT = Path(__file__).resolve().parents[2]


def config():
    return Config.model_validate(yaml.safe_load((ROOT / 'config.yaml').read_text()))


def endpoints(base):
    prototype = next(iter(config().llm.endpoints.values()))
    return {
        name: prototype.model_copy(update={
            'base_url': base + '/v1', 'model': name, 'api_key_env': 'L3_STUB_KEY',
            'structured_output': 'function_calling', 'extra_body': {},
        })
        for name in ('primary', 'fallback')
    }


def oneshot(timeout=1):
    return config().llm.scenes.preprocess.model_copy(update={
        'primary': 'primary', 'fallback': 'fallback', 'timeout_s': timeout,
    })
