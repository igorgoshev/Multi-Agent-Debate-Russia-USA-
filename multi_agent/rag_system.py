#!/usr/bin/env python3
"""
RAG System for AI Agent Negotiation
Retrieval-Augmented Generation system using Qdrant vector database

Requirements:
pip install qdrant-client sentence-transformers openai pydantic-ai asyncio

Usage:
python rag_system.py
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid

@dataclass
class RetrievalResult:
    """Represents a retrieved chunk with relevance score"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    
    def __str__(self):
        return f"Score: {self.score:.3f} | {self.content[:100]}..."

@dataclass
class RAGContext:
    """Context assembled from multiple retrieved chunks"""
    query: str
    retrieved_chunks: List[RetrievalResult]
    context_text: str
    sources: List[str]
    categories: List[str]
    total_tokens: int
    retrieval_timestamp: str

class QdrantRAGRetriever:
    """RAG retrieval system using Qdrant vector database"""
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "agent_chunks",
        embedding_model: str = "all-MiniLM-L6-v2",
        api_key: Optional[str] = None
    ):
        """
        Initialize RAG retriever
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Collection name in Qdrant
            embedding_model: Embedding model name (must match what was used for indexing)
            api_key: Qdrant API key (if needed)
        """
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Initialize Qdrant client
        if api_key:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=api_key)
        else:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Initialize embedding model
        print(f"🔄 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded. Dimensions: {self.embedding_dim}")
        
        # Verify connection and collection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Qdrant connection and collection exists"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name in collection_names:
                print(f"✅ Connected to Qdrant. Collection '{self.collection_name}' found.")
                
                # Get collection info
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                print(f"📊 Collection info: {collection_info.points_count} points, {collection_info.vectors_count} vectors")
            else:
                print(f"⚠️ Collection '{self.collection_name}' not found.")
                print(f"Available collections: {collection_names}")
                raise ValueError(f"Collection '{self.collection_name}' does not exist")
                
        except Exception as e:
            print(f"❌ Qdrant connection failed: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        embedding = self.embedding_model.encode([query])[0]
        return embedding.tolist()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score threshold
            filters: Additional filters (e.g., source, category, date)
            
        Returns:
            List of RetrievalResult objects
        """
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Build Qdrant filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    # Multiple values (OR condition)
                    for v in value:
                        conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))
                else:
                    # Single value
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            
            if conditions:
                qdrant_filter = Filter(should=conditions)
        
        # Search in Qdrant
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            score_threshold=score_threshold
        )
        
        # Convert to RetrievalResult objects
        results = []
        for result in search_results:
            retrieval_result = RetrievalResult(
                chunk_id=result.payload.get('chunk_id', str(result.id)),
                content=result.payload.get('content', ''),
                score=result.score,
                metadata=result.payload
            )
            results.append(retrieval_result)
        
        return results
    
    def build_context(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult],
        max_context_length: int = 2000,
        include_metadata: bool = True
    ) -> RAGContext:
        """
        Build context from retrieved chunks for LLM input
        
        Args:
            query: Original query
            retrieved_chunks: Retrieved chunks
            max_context_length: Maximum context length in characters
            include_metadata: Whether to include metadata in context
            
        Returns:
            RAGContext object
        """
        context_parts = []
        current_length = 0
        used_chunks = []
        
        for chunk in retrieved_chunks:
            chunk_text = chunk.content
            
            # Add metadata if requested
            if include_metadata:
                metadata_str = f"[Source: {chunk.metadata.get('source', 'Unknown')}, Category: {chunk.metadata.get('category', 'Unknown')}]"
                chunk_text = f"{metadata_str}\n{chunk_text}"
            
            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_text) > max_context_length and context_parts:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
            used_chunks.append(chunk)
        
        # Join context
        context_text = "\n\n---\n\n".join(context_parts)
        
        # Extract metadata
        sources = list(set(chunk.metadata.get('source', 'Unknown') for chunk in used_chunks))
        categories = list(set(chunk.metadata.get('category', 'Unknown') for chunk in used_chunks))
        
        # Estimate token count (rough approximation)
        total_tokens = len(context_text.split()) * 1.33
        
        return RAGContext(
            query=query,
            retrieved_chunks=used_chunks,
            context_text=context_text,
            sources=sources,
            categories=categories,
            total_tokens=int(total_tokens),
            retrieval_timestamp=datetime.now().isoformat()
        )
    
    def search_with_context(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        max_context_length: int = 2000,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> RAGContext:
        """
        Complete RAG search: retrieve chunks and build context
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score
            max_context_length: Maximum context length
            filters: Additional filters
            include_metadata: Include source metadata
            
        Returns:
            RAGContext with assembled information
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters
        )
        
        if not retrieved_chunks:
            print(f"⚠️ No relevant chunks found for query: '{query}'")
            return RAGContext(
                query=query,
                retrieved_chunks=[],
                context_text="No relevant information found.",
                sources=[],
                categories=[],
                total_tokens=0,
                retrieval_timestamp=datetime.now().isoformat()
            )
        
        # Build context
        context = self.build_context(
            query=query,
            retrieved_chunks=retrieved_chunks,
            max_context_length=max_context_length,
            include_metadata=include_metadata
        )
        
        return context

class MultiAgentRAGSystem:
    """RAG system managing multiple agents with separate knowledge bases"""
    
    def __init__(self, agent_configs: Dict[str, Dict[str, Any]]):
        """
        Initialize multi-agent RAG system
        
        Args:
            agent_configs: Dictionary mapping agent names to their Qdrant configurations
                Example: {
                    "Agent_USA": {"host": "localhost", "port": 6333, "collection": "usa_chunks"},
                    "Agent_Russia": {"host": "localhost", "port": 6334, "collection": "russia_chunks"}
                }
        """
        self.agent_configs = agent_configs
        self.retrievers = {}
        
        # Initialize retrievers for each agent
        for agent_name, config in agent_configs.items():
            print(f"🔄 Initializing retriever for {agent_name}...")
            retriever = QdrantRAGRetriever(
                qdrant_host=config.get("host", "localhost"),
                qdrant_port=config.get("port", 6333),
                collection_name=config.get("collection", f"{agent_name.lower()}_chunks"),
                embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
                api_key=config.get("api_key")
            )
            self.retrievers[agent_name] = retriever
            print(f"✅ Retriever for {agent_name} ready")
    
    def search_for_agent(
        self,
        agent_name: str,
        query: str,
        **search_kwargs
    ) -> RAGContext:
        """Search knowledge base for specific agent"""
        if agent_name not in self.retrievers:
            raise ValueError(f"Agent '{agent_name}' not found. Available agents: {list(self.retrievers.keys())}")
        
        retriever = self.retrievers[agent_name]
        return retriever.search_with_context(query, **search_kwargs)
    
    def search_all_agents(
        self,
        query: str,
        **search_kwargs
    ) -> Dict[str, RAGContext]:
        """Search knowledge bases for all agents"""
        results = {}
        
        for agent_name in self.retrievers:
            print(f"🔍 Searching for {agent_name}...")
            results[agent_name] = self.search_for_agent(agent_name, query, **search_kwargs)
        
        return results
    
    def compare_agent_knowledge(
        self,
        query: str,
        agents: Optional[List[str]] = None,
        **search_kwargs
    ) -> Dict[str, Any]:
        """Compare what different agents know about a topic"""
        if agents is None:
            agents = list(self.retrievers.keys())
        
        results = {}
        for agent_name in agents:
            if agent_name in self.retrievers:
                results[agent_name] = self.search_for_agent(agent_name, query, **search_kwargs)
        
        # Analyze differences
        analysis = {
            "query": query,
            "agent_results": results,
            "comparison": self._analyze_knowledge_differences(results)
        }
        
        return analysis
    
    def _analyze_knowledge_differences(self, results: Dict[str, RAGContext]) -> Dict[str, Any]:
        """Analyze differences between agents' knowledge"""
        analysis = {
            "total_chunks_found": {agent: len(ctx.retrieved_chunks) for agent, ctx in results.items()},
            "unique_sources": {},
            "common_themes": set(),
            "agent_specific_info": {}
        }
        
        # Extract unique sources per agent
        for agent_name, context in results.items():
            analysis["unique_sources"][agent_name] = context.sources
        
        # Find common categories
        all_categories = []
        for context in results.values():
            all_categories.extend(context.categories)
        
        # Count category occurrences
        from collections import Counter
        category_counts = Counter(all_categories)
        analysis["common_categories"] = dict(category_counts)
        
        return analysis

def test_rag_system():
    """Test the RAG system with sample queries"""
    
    print("🧪 Testing RAG System")
    print("=" * 50)
    
    # Configuration for your setup
    agent_configs = {
        "Agent_USA": {
            "host": "localhost",
            "port": 6333,  # First Qdrant instance
            "collection": "usa_collection",
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "Agent_Russia": {
            "host": "localhost", 
            "port": 6334,  # Second Qdrant instance (or different collection)
            "collection": "russia_collection",
            "embedding_model": "all-MiniLM-L6-v2"
        }
    }
    
    # Initialize multi-agent RAG system
    rag_system = MultiAgentRAGSystem(agent_configs)
    
    # Test queries
    test_queries = [
        "Ukraine war negotiations",
        "Economic sanctions impact",
        "NATO expansion policy",
        "Energy security concerns",
        "Diplomatic relations"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        print("-" * 30)
        
        try:
            # Search for both agents
            usa_context = rag_system.search_for_agent("Agent_USA", query, top_k=3)
            russia_context = rag_system.search_for_agent("Agent_Russia", query, top_k=3)
            
            print(f"USA Agent found {len(usa_context.retrieved_chunks)} relevant chunks")
            if usa_context.retrieved_chunks:
                print(f"  Top result: {usa_context.retrieved_chunks[0]}")
            
            print(f"Russia Agent found {len(russia_context.retrieved_chunks)} relevant chunks")
            if russia_context.retrieved_chunks:
                print(f"  Top result: {russia_context.retrieved_chunks[0]}")
            
            # Compare knowledge
            comparison = rag_system.compare_agent_knowledge(query)
            print(f"Knowledge comparison: {comparison['comparison']['total_chunks_found']}")
            
        except Exception as e:
            print(f"❌ Error testing query '{query}': {e}")
    
    print(f"\n✅ RAG system testing complete!")

if __name__ == "__main__":
    # Run tests
    test_rag_system()