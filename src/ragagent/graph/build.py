"""所有回答路径汇到postprocess；context只存运行依赖，SQLite只存状态。"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ragagent.graph.deps import Deps
from ragagent.graph.nodes.fallback import fallback_node
from ragagent.graph.nodes.generate import generate_node
from ragagent.graph.nodes.postprocess import postprocess_node
from ragagent.graph.nodes.routes import route_after_preprocess, route_after_rerank, route_after_rewrite
from ragagent.graph.state import RAGState


def _default_nodes():
    # 延迟导入：L2交付前可注入占位节点测试全部条件边，之后默认始终使用真实节点。
    from ragagent.graph.nodes.preprocess import preprocess as preprocess_node
    from ragagent.graph.nodes.retrieve import retrieve as retrieve_node
    from ragagent.graph.nodes.rerank import rerank as rerank_node
    from ragagent.graph.nodes.build_context import build_context as build_context_node
    from ragagent.graph.nodes.rewrite import rewrite as rewrite_node
    from ragagent.graph.nodes.clarify import clarify as clarify_node
    return dict(preprocess=preprocess_node, retrieve=retrieve_node, rerank=rerank_node,
                build_context=build_context_node, rewrite=rewrite_node, clarify=clarify_node)


def build_graph(checkpointer=None, *, nodes=None):
    implementations = _default_nodes() if nodes is None else dict(nodes)
    implementations = dict(generate=generate_node, fallback=fallback_node, postprocess=postprocess_node) | implementations
    required = {'preprocess', 'retrieve', 'rerank', 'build_context', 'generate', 'rewrite', 'fallback', 'clarify', 'postprocess'}
    if set(implementations) != required:
        raise ValueError('nodes 必须提供全部规定节点，不能省略回答路径')
    graph = StateGraph(RAGState, context_schema=Deps)
    for name, node in implementations.items():
        graph.add_node(name, node)
    graph.add_edge(START, 'preprocess')
    graph.add_conditional_edges('preprocess', route_after_preprocess, {
        'retrieve': 'retrieve', 'generate': 'generate', 'clarify': 'clarify'})
    graph.add_edge('retrieve', 'rerank')
    graph.add_conditional_edges('rerank', route_after_rerank, {
        'build_context': 'build_context', 'rewrite': 'rewrite', 'fallback': 'fallback'})
    graph.add_conditional_edges('rewrite', route_after_rewrite, {'retrieve': 'retrieve', 'fallback': 'fallback'})
    graph.add_edge('build_context', 'generate')
    for name in ('generate', 'fallback', 'clarify'):
        graph.add_edge(name, 'postprocess')
    graph.add_edge('postprocess', END)
    return graph.compile(checkpointer=checkpointer)
