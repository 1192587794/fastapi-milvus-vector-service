import math

from utils.demo_embedding import DemoTextEmbedding


def test_encode_returns_expected_dimension() -> None:
    embedding = DemoTextEmbedding(dimension=64)
    vector = embedding.encode("milvus fastapi template")
    assert len(vector) == 64


def test_encode_is_deterministic() -> None:
    embedding = DemoTextEmbedding(dimension=64)
    first = embedding.encode("same input")
    second = embedding.encode("same input")
    assert first == second


def test_non_empty_text_is_l2_normalized() -> None:
    embedding = DemoTextEmbedding(dimension=64)
    vector = embedding.encode("vector search")
    norm = math.sqrt(sum(value * value for value in vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-9)


def test_empty_text_returns_zero_vector() -> None:
    embedding = DemoTextEmbedding(dimension=64)
    vector = embedding.encode("   ")
    assert vector == [0.0] * 64
