"""M2：真实L2节点、SQLite、Prompt、生成与后处理，HTTP编码/LLM桩。"""
import pytest
from tests.integration.helpers import KEYS, api_client, config, deps, events, preprocess_script, rows


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in KEYS:
        monkeypatch.setenv(name, 'test')


def test_m2_real_nodes_http_stubs_end_to_end_sse(tmp_path, llm_url, embed_url):
    cfg = config(tmp_path, llm_url)
    dep = deps(cfg, embed_url)
    preprocess_script(llm_url)
    with api_client(cfg, dep) as client:
        wire = events(client.post('/api/chat', json={'question': rows()[2]['text'], 'session_id': 'm2'}))
        types = [e['type'] for e in wire]
        assert types[0] == 'progress'
        assert types.index('meta') < types.index('token') < types.index('final')
        assert types[-1] == 'final' and 'error' not in types
        assert {e.get('stage') for e in wire} >= {'preprocessing', 'retrieving', 'reranking', 'building_context', 'generating'}
        saved = client.portal.call(client.app.state.graph.aget_state, {'configurable': {'thread_id': 'm2'}})
        assert len(saved.values['messages']) == 2
        assert saved.values['gen_meta']['finish_reason'] == 'stop'
        assert saved.values['gen_meta']['usage_estimated'] is False
        assert saved.values['gen_meta']['usage']['output_tokens'] > 0
        assert saved.values['gen_meta']['post']['citation_support'] is not None
        assert client.get('/metrics').status_code == 200
        assert 'rag_requests_total{route="rag",status="ok"} 1.0' in client.get('/metrics').text
        metrics = client.get('/metrics').text
        assert 'rag_stage_seconds_count{stage="ttft"} 1.0' in metrics
        assert 'rag_llm_calls_total{endpoint="small",outcome="ok",scene="preprocess"} 1.0' in metrics
        assert 'rag_llm_calls_total{endpoint="main",outcome="ok",scene="generate"} 1.0' in metrics
        assert dep.milvus.calls
