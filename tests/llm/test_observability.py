"""D-013 B：旧接口、全局埋点与 RunnableConfig 身份保持一致。"""
import asyncio
import inspect
from dataclasses import fields, replace

import pytest
from langgraph.graph import END, START, StateGraph

from ragagent.graph.deps import Deps
from ragagent.llm import observability
from ragagent.llm.observability import current_message_id, metrics_scope, structured_scene
from ragagent.llm.stream import stream_answer
from ragagent.llm.structured import structured_call
from ragagent.preprocess.schema import Preprocessed
from tests.llm.support import config


def test_original_public_signatures_have_no_new_injection_parameters():
    assert str(inspect.signature(structured_call)) == '(scene, eps, schema, messages)'
    assert list(inspect.signature(stream_answer).parameters) == ['pkg', 'scene', 'eps', 'sems', 'emit']
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
               for p in inspect.signature(stream_answer).parameters.values())
    names = {field.name for field in fields(Deps)}
    assert not names & {'metrics', 'message_id'}
    with pytest.raises(TypeError):
        structured_call(None, None, None, None, metrics=None)
    with pytest.raises(TypeError):
        stream_answer(None, None, None, None, None, clients={})


def test_deps_reads_global_metrics_but_does_not_accept_injection(monkeypatch):
    metrics = object()
    monkeypatch.setattr(observability, 'METRICS', metrics)
    dep = Deps(config(), len)
    assert dep.metrics is metrics
    with pytest.raises(TypeError):
        Deps(config(), len, metrics=metrics)
    with pytest.raises(TypeError):
        replace(dep, metrics=None)
    with pytest.raises(AttributeError):
        dep.metrics = None
    monkeypatch.setattr(observability, 'METRICS', None)
    assert dep.metrics is None


def test_metrics_scope_nesting_and_exception_restore(monkeypatch):
    previous, active = object(), object()
    monkeypatch.setattr(observability, 'METRICS', previous)
    with metrics_scope(active):
        assert observability.get_metrics() is active
        with pytest.raises(RuntimeError), metrics_scope(None):
            assert observability.get_metrics() is None
            raise RuntimeError('synthetic')
        assert observability.get_metrics() is active
    assert observability.get_metrics() is previous


def test_standalone_metadata_and_fixed_scene_names():
    assert current_message_id() is None
    assert structured_scene(Preprocessed) == 'preprocess'
    assert structured_scene(type('Rewrite', (), {})) == 'rewrite'
    assert structured_scene(type('private_unapproved_label', (), {})) == 'eval'


def test_runnable_metadata_isolated_across_concurrent_tasks():
    async def run():
        async def node(state):
            await asyncio.sleep(0)
            return {'id': current_message_id(), 'scene': structured_scene(Preprocessed)}
        graph = StateGraph(dict)
        graph.add_node('rewrite', node)
        graph.add_edge(START, 'rewrite')
        graph.add_edge('rewrite', END)
        app = graph.compile()
        values = await asyncio.gather(*(app.ainvoke({}, config={'metadata': {'message_id': mid}})
                                      for mid in ('synthetic-one', 'synthetic-two')))
        assert values == [{'id': 'synthetic-one', 'scene': 'rewrite'},
                          {'id': 'synthetic-two', 'scene': 'rewrite'}]
        assert current_message_id() is None
    asyncio.run(run())
