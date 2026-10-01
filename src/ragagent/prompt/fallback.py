from pathlib import Path
from types import SimpleNamespace

from ragagent.prompt.env import build_env


def render_fallback(state, cfg, *, env=None) -> str:
    env = env if env is not None else build_env(Path(cfg.prompt.dir) / cfg.prompt.version)
    candidates = state.get('candidates', [])
    sources = []
    for item in sorted(candidates, key=lambda c: c.get('score', 0), reverse=True)[:cfg.prompt.fallback_sources]:
        sources.append(SimpleNamespace(title=item.get('title', ''), section_path=item.get('section_path', ''),
                                       page_start=item.get('page_start', -1)))
    return env.get_template('fallback.j2').render(sources=sources)
