import asyncio

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from ragagent.graph.deps import Deps, emit
from tests.llm.support import config


def test_runtime_context_and_custom_writer():
    async def run():
        deps = Deps(cfg=config(), count=len)
        async def node(state: dict, runtime: Runtime[Deps]):
            assert runtime.context is deps
            emit(runtime, {'event': 'progress', 'data': {'node': 'probe'}})
            return {'count': runtime.context.count('合成')}
        graph = StateGraph(dict, context_schema=Deps)
        graph.add_node('probe', node)
        graph.add_edge(START, 'probe')
        graph.add_edge('probe', END)
        app = graph.compile()
        events = [e async for e in app.astream({}, context=deps, stream_mode='custom')]
        assert events == [{'event': 'progress', 'data': {'node': 'probe'}}]
        assert await app.ainvoke({}, context=deps) == {'count': 2}
    asyncio.run(run())


def test_write_override_never_double_emits():
    events = []
    deps = Deps(cfg=config(), count=len, write=events.append)
    runtime = Runtime(context=deps, stream_writer=lambda _: (_ for _ in ()).throw(AssertionError('double write')))
    emit(runtime, {'event': 'token', 'data': {'text': '合成'}})
    assert events == [{'event': 'token', 'data': {'text': '合成'}}]
