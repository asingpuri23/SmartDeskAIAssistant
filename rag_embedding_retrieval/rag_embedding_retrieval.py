import json
from sentence_transformers import SentenceTransformer
from sentence_transformers import SparseEncoder
import fastembed
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance,
                                  SparseVector,
                                  PointStruct,
                                  QueryRequest,
                                  SparseIndexParams,
                                  SparseVectorParams,
                                  VectorParams,
                                  ScoredPoint)
import numpy as np
from pathlib import Path


class RAGRetriever:
    
    def make_points(self, df:pd.DataFrame) -> list[PointStruct]:
          sparse_vectors = df['sparse_embedding'].tolist()
          product_texts = df['Combined_text'].tolist()
          dense_vectors = df['dense_embedding'].tolist()
          rows = df.to_dict(orient='records')
          #print(len(sparse_vectors), len(product_texts), len(dense_vectors))
          #print(len(rows))
          points = []
          for idx, (text, sparse_vector, dense_vector) in enumerate(zip(product_texts, sparse_vectors, dense_vectors)):
            #print("idx ",idx)
            sparse_vector = SparseVector(indices=sparse_vector.indices.tolist(), values = sparse_vector.values.tolist())
            point = PointStruct(id = idx, payload={
                'text':text
            },
                                vector={
                                    "text-sparse": sparse_vector,
                                    "text-dense":dense_vector
                                })
            points.append(point)
    
          return points
    
    def make_sparse_embedding(self, texts):
            return list(self.model_sparse.embed(texts, batch_size=8))
    
    def make_dense_embedding(self, texts):
            embeddings = list(self.model_dense.embed(texts))
            return embeddings
    
    def retrieve(self, query:str, top_k:int = 10):
            search_results = self.search(query, top_k)
            rank_lists = self.rank_list(search_results)
            rrf_rank_list = self.rrf(rank_lists)
            fused_records = self.client.retrieve(collection_name=self.collection_name, ids=[item[0] for item in rrf_rank_list])
            record_by_id = {r.id: r for r in fused_records}
            return record_by_id
    
    def search(self, query:str, top_k:int = 10):
          query_sparse_vectors = self.make_sparse_embedding([query])
          query_dense_vector = self.make_dense_embedding([query])
          #print(query_sparse_vectors)
          #print(query_dense_vector)
          search_results = self.client.query_batch_points(collection_name=self.collection_name,
                                                    requests=[QueryRequest(query=query_dense_vector[0].tolist(),
                                                                            using="text-dense",
                                                                            limit=top_k,
                                                                            with_payload=True),
                                                              QueryRequest(query=SparseVector(indices=query_sparse_vectors[0].indices.tolist(),
                                                                                              values=query_sparse_vectors[0].values.tolist()),
                                                                            using='text-sparse',
                                                                            limit=top_k,
                                                                            with_payload=True)])
          print(search_results)
          return [search_results[0].points, search_results[1].points]
    
    
    def rank_list(self, search_results):
          dense_results, sparse_results = search_results[0], search_results[1]
          ranklist_dense = []
          ranklist_sparse = []
          for i, point in enumerate(dense_results):
            ranktuple = tuple([point.id, i+1])
            ranklist_dense.append(ranktuple)
          for i, point in enumerate(sparse_results):
            ranktuple = tuple([point.id, i+1])
            ranklist_sparse.append(ranktuple)
          ranklists = [ranklist_dense, ranklist_sparse]
          return ranklists
    
    def rrf(self, rank_lists, k=60, default_rank=1000):
          all_items = set(item for rank_list in rank_lists for item, _ in rank_list)
          item_to_index = {item:idx for idx, item in enumerate(all_items)}
    
          rank_matrix = np.full((len(all_items), len(rank_lists)), default_rank)
    
          for list_idx, rank_list in enumerate(rank_lists):
            for item, rank in rank_list:
              rank_matrix[item_to_index[item], list_idx] = rank
    
          rrf_scores = np.sum(1.0/ (k+rank_matrix), axis=1)
    
          sorted_indices = np.argsort(-rrf_scores)
          sorted_items = [(list(item_to_index.keys())[idx], rrf_scores[idx]) for idx in sorted_indices]
    
          return sorted_items
    
    
    def hybrid_search(self, query_text:str, top_k:int = 10):
            search_results = self.search(query_text, top_k)
            rank_lists = self.rank_list(search_results)
            rrf_rank_list = self.rrf(rank_lists)
            fused_records = self.client.retrieve(collection_name=self.collection_name, ids=[item[0] for item in rrf_rank_list])
            record_by_id = {r.id: r for r in fused_records}
    
            print(f"\n\nHybrid search results for {query_text}")
            print("===================================================================")
    
            document_list = []
            for rank, (item_id, score) in enumerate(rrf_rank_list, 1):
                record = record_by_id[item_id]
                title = record.payload['text'].split('\n')[0][:75]
                in_dense = any(id == item_id for id, _ in rank_lists[0])
                in_sparse = any(id == item_id for id, _ in rank_lists[1])
                source = "BOTH" if (in_dense and in_sparse) else ("Dense only" if in_dense else "Sparse only")
                print(f"  {rank:>2}.  [{source:<12}]   {title}")
                document_list.append(record.payload['text'])
    
            return document_list
    
    def __init__(self):
        print("Initializing RAGRetriever...")
        parent_dir = Path(__file__).resolve().parent.parent
        with open(parent_dir / "data/sample.json", "r") as f:
            data = json.load(f)
        print(type(data))

        self.df = pd.DataFrame( columns=["Combined_text", "sparse_Embedding", "dense_Embedding"])
        self.df["Combined_text"] = data

        self.model_sparse = SparseTextEmbedding(model_name="prithvida/Splade_PP_en_v1", batch_size=8)
        self.model_dense = TextEmbedding(model_name="BAAI/bge-large-en-v1.5", batch_size=8 )

        print("models downloaded and loaded successfully...")

        self.df['sparse_embedding'] = self.make_sparse_embedding(data)
        self.df['dense_embedding'] = self.make_dense_embedding(data)

        print("creating Qdrant client and collection...")
        self.client = QdrantClient(":memory:")
        self.collection_name = "hr_it_policies_qa_collection"
        self.client.create_collection( self.collection_name, vectors_config={"text-dense": VectorParams(size=1024, distance=Distance.COSINE)},
                                sparse_vectors_config={"text-sparse":SparseVectorParams(index=SparseIndexParams(on_disk=False))})

        self.points = self.make_points(self.df)
        print(len(self.points))

        print("upserting points to Qdrant collection...")
        self.client.upsert(collection_name=self.collection_name, points=self.points)


    

#testing the RAGRetriever class
ragRetriever = RAGRetriever()
ragRetriever.hybrid_search("How many leaves are available per year?", top_k=5)
