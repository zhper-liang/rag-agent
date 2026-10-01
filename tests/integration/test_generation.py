"""真实生成/后处理节点：不完整响应也只写一轮历史，usage缺失时估算。"""
import asyncio

import pytest
from langchain_core.messages import AIMessageChunk

from ragagent.api.app import initial_state
from ragagent.graph.build import build_graph
from ragagent.graph.deps import Deps
from ragagent.preprocess.schema import Preprocessed
from tests.integration.helpers import config


@pytest.mark.parametrize('case,status', [('normal', 'ok'), ('length', 'truncated'), ('error', 'interrupted')])
def test_generation_status_usage_and_saved_plain(tmp_path, case, status):
    async def run():
        cfg, wire = config(tmp_path), []
        sem = asyncio.Semaphore(1)
        closed = False

        class Client:
            async def astream(self, messages):
                nonlocal closed
                try:
                    yield AIMessageChunk(content='合成😀')
                    if case == 'error':
                        raise ConnectionError('private provider text must not escape')
                    yield AIMessageChunk(content='回答', response_metadata={'finish_reason': 'length' if case == 'length' else 'stop'})
                finally:
                    closed = True

        async def structured(scene, eps, schema, messages):
            return Preprocessed(intent='chat', standalone_question='合成问题'), scene.primary

        dep = Deps(cfg=cfg, count=len, structured_call=structured, write=wire.append,
                   clients={('small', 'chat'): Client()}, sems={'small': sem})
        state = await build_graph().ainvoke(initial_state('合成问题', None), context=dep)
        meta, final = state['gen_meta'], wire[-1]
        assert meta['status'] == final['status'] == status
        assert final['type'] == 'final' and not any(e['type'] == 'error' for e in wire)
        assert state['answer'] == ''.join(e['text'] for e in wire if e['type'] == 'token')
        assert meta['usage_estimated'] is True and meta['usage']['output_tokens'] == len(state['answer'])
        assert meta['usage']['input_tokens'] == meta['est_input_tokens']
        assert meta['attempts'] == [] and not meta['fallback_used']
        assert len(state['messages']) == 2
        ai = state['messages'][-1]
        expected = state['answer'] + (cfg.postprocess.incomplete_suffix if status != 'ok' else '')
        assert ai.content == state['answer'] and ai.additional_kwargs['plain'] == expected
        assert ai.additional_kwargs['message_id'] == final['message_id']
        assert ai.additional_kwargs['status'] == status
        assert closed and not sem.locked()
        assert 'private provider text' not in str(wire) + str(meta)
    asyncio.run(run())
