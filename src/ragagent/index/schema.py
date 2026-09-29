"""Milvus 字段、BM25 Function 与索引定义（向量化建库.md §5.2）。"""

from pymilvus import DataType, Function, FunctionType, MilvusClient


def create_collection(client: MilvusClient, name: str, dim: int) -> None:
    """dim 取 index.dim。"""
    s = client.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field("chunk_id", DataType.VARCHAR, max_length=64, is_primary=True)
    s.add_field("doc_id", DataType.VARCHAR, max_length=32)
    s.add_field("version", DataType.INT32)
    s.add_field("seq", DataType.INT32)
    s.add_field("tokens", DataType.INT32)
    s.add_field("section_idx", DataType.INT32)
    s.add_field("section_tokens", DataType.INT32)
    s.add_field("type", DataType.VARCHAR, max_length=16)
    s.add_field("text", DataType.VARCHAR, max_length=65535)
    s.add_field("embed_text", DataType.VARCHAR, max_length=65535,
                enable_analyzer=True, analyzer_params={"type": "chinese"})
    s.add_field("content_hash", DataType.VARCHAR, max_length=64)
    s.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
    s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    s.add_field("title", DataType.VARCHAR, max_length=512)
    s.add_field("section_path", DataType.VARCHAR, max_length=2048)
    s.add_field("source_path", DataType.VARCHAR, max_length=1024)
    s.add_field("file_type", DataType.VARCHAR, max_length=16)
    s.add_field("page_start", DataType.INT32)
    s.add_field("page_end", DataType.INT32)
    s.add_field("image_path", DataType.VARCHAR, max_length=1024)
    s.add_field("prev_id", DataType.VARCHAR, max_length=64)
    s.add_field("next_id", DataType.VARCHAR, max_length=64)
    s.add_field("image_ids", DataType.ARRAY, element_type=DataType.VARCHAR,
                max_capacity=32, max_length=64)
    s.add_field("related_ids", DataType.ARRAY, element_type=DataType.VARCHAR,
                max_capacity=32, max_length=64)
    s.add_field("searchable", DataType.BOOL)
    s.add_field("meta", DataType.JSON)

    # BM25：embed_text 分词后自动生成 sparse，写入时不需要提供
    s.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                            input_field_names=["embed_text"],
                            output_field_names=["sparse"]))

    idx = client.prepare_index_params()
    idx.add_index("dense", index_type="DISKANN", metric_type="COSINE")
    idx.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    idx.add_index("doc_id", index_type="INVERTED")
    idx.add_index("type", index_type="BITMAP")
    idx.add_index("file_type", index_type="BITMAP")
    idx.add_index("searchable", index_type="BITMAP")
    client.create_collection(name, schema=s, index_params=idx)
