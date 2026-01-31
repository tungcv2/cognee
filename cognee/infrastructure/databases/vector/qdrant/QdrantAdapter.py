import asyncio
from typing import List, Optional, Any
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.databases.exceptions import MissingQueryParameterError
from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..vector_db_interface import VectorDBInterface
from ..utils import normalize_distances

class QdrantAdapter(VectorDBInterface):
    name = "Qdrant"
    url: str
    api_key: str
    connection: AsyncQdrantClient = None

    def __init__(self, url: Optional[str], api_key: Optional[str], embedding_engine: EmbeddingEngine):
        self.url = url or "http://localhost:6333"
        self.api_key = api_key
        self.embedding_engine = embedding_engine
        self.VECTOR_DB_LOCK = asyncio.Lock()

    async def get_connection(self) -> AsyncQdrantClient:
        if self.connection is None:
            self.connection = AsyncQdrantClient(url=self.url, api_key=self.api_key)
        return self.connection

    async def embed_data(self, data: List[str]) -> List[List[float]]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        client = await self.get_connection()
        collections = await client.get_collections()
        return any(c.name == collection_name for c in collections.collections)

    async def create_collection(self, collection_name: str, payload_schema: Any = None):
        async with self.VECTOR_DB_LOCK:
            client = await self.get_connection()
            if not await self.has_collection(collection_name):
                vector_size = self.embedding_engine.get_vector_size()
                await client.create_collection(
                    collection_name = collection_name,
                    vectors_config = models.VectorParams(size = vector_size, distance = models.Distance.COSINE),
                )

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        await self.create_collection(collection_name)
        client = await self.get_connection()
        texts = [DataPoint.get_embeddable_data(dp) for dp in data_points]
        embeddings = await self.embed_data(texts)
        points = [
            models.PointStruct(
                id = str(dp.id),
                vector = embeddings[i],
                payload = dp.model_dump() if hasattr(dp, "model_dump") else dp.__dict__
            ) for i, dp in enumerate(data_points)
        ]
        await client.upsert(collection_name = collection_name, points = points)

    async def retrieve(self, collection_name: str, data_point_ids: List[str]):
        client = await self.get_connection()
        points = await client.retrieve(collection_name = collection_name, ids = data_point_ids, with_payload = True)
        return [ScoredResult(id = parse_id(p.id), payload = p.payload, score = 0) for p in points]

    async def search(self, collection_name: str, query_text: str = None, query_vector: List[float] = None, limit: int = 15, with_vector: bool = False):
        if query_text is None and query_vector is None:
            raise MissingQueryParameterError()
        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]
        client = await self.get_connection()
        results = await client.search(collection_name = collection_name, query_vector = query_vector, limit = limit or 15, with_payload = True, with_vectors = with_vector)
        if not results: return []
        scored_results = [{"id": parse_id(r.id), "payload": r.payload, "_distance": 1.0 - r.score} for r in results]
        normalized_scores = normalize_distances(scored_results)
        return [ScoredResult(id = scored_results[i]["id"], payload = scored_results[i]["payload"], score = normalized_scores[i]) for i in range(len(results))]

    async def batch_search(self, collection_name: str, query_texts: List[str], limit: int = 5, with_vectors: bool = False):
        query_vectors = await self.embed_data(query_texts)
        requests = [models.SearchRequest(vector = v, limit = limit or 5, with_payload = True, with_vectors = with_vectors) for v in query_vectors]
        client = await self.get_connection()
        batch_results = await client.search_batch(collection_name = collection_name, requests = requests)
        output = []
        for results in batch_results:
            if not results:
                output.append([])
                continue
            scored_results = [{"id": parse_id(r.id), "payload": r.payload, "_distance": 1.0 - r.score} for r in results]
            normalized_scores = normalize_distances(scored_results)
            output.append([ScoredResult(id = scored_results[i]["id"], payload = scored_results[i]["payload"], score = normalized_scores[i]) for i in range(len(results))])
        return output

    async def delete_data_points(self, collection_name: str, data_point_ids: List[str]):
        client = await self.get_connection()
        await client.delete(collection_name = collection_name, points_selector = models.PointIdsList(points = data_point_ids))

    async def prune(self):
        client = await self.get_connection()
        collections = await client.get_collections()
        for c in collections.collections:
            await client.delete_collection(collection_name = c.name)
