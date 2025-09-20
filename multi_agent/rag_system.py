#!/usr/bin/env python3
"""
Improved RAG System for AI Agent Negotiation
Enhanced Retrieval-Augmented Generation system with adaptive context and multi-perspective retrieval

Requirements:
pip install qdrant-client sentence-transformers numpy
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from collections import Counter
import logging

logger = logging.getLogger(__name__)


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
    perspective_type: str = "standard"  # standard, supporting, opposing, balanced


class EnhancedQdrantRAGRetriever:
    """Enhanced RAG retrieval system with adaptive context and multi-perspective support"""

    def __init__(
            self,
            qdrant_host: str = "localhost",
            qdrant_port: int = 6333,
            collection_name: str = "agent_chunks",
            embedding_model: str = "all-MiniLM-L6-v2",
            api_key: Optional[str] = None
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        if api_key:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=api_key)
        else:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        self.query_cache = {}
        self.cache_size = 50

        self._verify_connection()

    def _verify_connection(self):
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name in collection_names:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                logger.info(
                    f"Connected to Qdrant. Collection '{self.collection_name}' found with {collection_info.points_count} points")
            else:
                logger.warning(f"Collection '{self.collection_name}' not found. Available: {collection_names}")
                raise ValueError(f"Collection '{self.collection_name}' does not exist")

        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query with caching"""
        if query in self.query_cache:
            return self.query_cache[query]

        embedding = self.embedding_model.encode([query])[0]
        embedding_list = embedding.tolist()

        if len(self.query_cache) >= self.cache_size:
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]

        self.query_cache[query] = embedding_list
        return embedding_list

    def retrieve_with_keywords(
            self,
            query: str,
            personality_keywords: Optional[List[str]] = None,
            phase_keywords: Optional[List[str]] = None,
            top_k: int = 5,
            score_threshold: float = 0.7,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Enhanced retrieval with personality and phase-specific keyword boosting"""

        query_embedding = self.embed_query(query)

        qdrant_filter = self._build_enhanced_filter(filters, personality_keywords, phase_keywords)

        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k * 2,  # Get more results for reranking
            score_threshold=score_threshold * 0.8  # Lower threshold for reranking
        )

        results = []
        for result in search_results:
            retrieval_result = RetrievalResult(
                chunk_id=result.payload.get('chunk_id', str(result.id)),
                content=result.payload.get('content', ''),
                score=result.score,
                metadata=result.payload
            )
            results.append(retrieval_result)

        if personality_keywords or phase_keywords:
            results = self._rerank_with_keywords(results, personality_keywords, phase_keywords)

        return results[:top_k]

    def _build_enhanced_filter(
            self,
            base_filters: Optional[Dict[str, Any]],
            personality_keywords: Optional[List[str]],
            phase_keywords: Optional[List[str]]
    ) -> Optional[Filter]:
        """Build enhanced Qdrant filter"""
        conditions = []

        # Add base filters
        if base_filters:
            for key, value in base_filters.items():
                if isinstance(value, list):
                    for v in value:
                        conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))
                else:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        if personality_keywords:
            for keyword in personality_keywords:
                conditions.append(FieldCondition(key="keywords", match=MatchValue(value=keyword)))

        return Filter(should=conditions) if conditions else None

    def _rerank_with_keywords(
            self,
            results: List[RetrievalResult],
            personality_keywords: Optional[List[str]],
            phase_keywords: Optional[List[str]]
    ) -> List[RetrievalResult]:
        """Rerank results based on keyword relevance"""
        all_keywords = []
        if personality_keywords:
            all_keywords.extend(personality_keywords)
        if phase_keywords:
            all_keywords.extend(phase_keywords)

        if not all_keywords:
            return results

        # Calculate keyword boost scores
        for result in results:
            content_lower = result.content.lower()
            keyword_score = 0

            for keyword in all_keywords:
                if keyword.lower() in content_lower:
                    # Boost based on keyword frequency
                    frequency = content_lower.count(keyword.lower())
                    keyword_score += frequency * 0.1

            # Combine original score with keyword boost
            result.score = result.score + keyword_score

        # Sort by enhanced score
        return sorted(results, key=lambda x: x.score, reverse=True)

    def build_adaptive_context(
            self,
            query: str,
            retrieved_chunks: List[RetrievalResult],
            agent_personality: str,
            negotiation_phase: str,
            max_context_length: Optional[int] = None,
            include_metadata: bool = True
    ) -> RAGContext:
        """Build context adapted to agent personality and negotiation phase"""

        # Determine adaptive context length
        if max_context_length is None:
            phase_lengths = {
                "opening": 1500,
                "information_exchange": 2000,
                "bargaining": 2500,
                "problem_solving": 2200,
                "decision_making": 1800,
                "closing": 1200
            }
            max_context_length = phase_lengths.get(negotiation_phase, 2000)

        # Prioritize chunks based on personality
        prioritized_chunks = self._prioritize_by_personality(retrieved_chunks, agent_personality)

        # Build context with prioritized chunks
        context_parts = []
        current_length = 0
        used_chunks = []

        for chunk in prioritized_chunks:
            chunk_text = chunk.content

            # Add metadata if requested
            if include_metadata:
                metadata_str = f"[Source: {chunk.metadata.get('source', 'Unknown')}, Relevance: {chunk.score:.2f}]"
                chunk_text = f"{metadata_str}\n{chunk_text}"

            # Check length constraint
            if current_length + len(chunk_text) > max_context_length and context_parts:
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)
            used_chunks.append(chunk)

        # Join context with phase-appropriate separators
        if negotiation_phase in ["bargaining", "problem_solving"]:
            context_text = "\n\n--- RELEVANT INFORMATION ---\n\n".join(context_parts)
        else:
            context_text = "\n\n".join(context_parts)

        # Extract metadata
        sources = list(set(chunk.metadata.get('source', 'Unknown') for chunk in used_chunks))
        categories = list(set(chunk.metadata.get('category', 'Unknown') for chunk in used_chunks))

        # Estimate token count
        total_tokens = len(context_text.split()) * 1.33

        return RAGContext(
            query=query,
            retrieved_chunks=used_chunks,
            context_text=context_text,
            sources=sources,
            categories=categories,
            total_tokens=int(total_tokens),
            retrieval_timestamp=datetime.now().isoformat(),
            perspective_type="adaptive"
        )

    def _prioritize_by_personality(
            self,
            chunks: List[RetrievalResult],
            personality: str
    ) -> List[RetrievalResult]:
        """Prioritize chunks based on agent personality"""

        personality_keywords = {
            "hawk": ["security", "threat", "defense", "military", "protection", "deterrence"],
            "dove": ["cooperation", "peace", "dialogue", "compromise", "mutual", "partnership"],
            "economist": ["economic", "trade", "market", "business", "investment", "cost"],
            "legalist": ["law", "treaty", "legal", "agreement", "binding", "international"],
            "innovator": ["innovation", "technology", "creative", "new", "experimental", "pilot"]
        }

        keywords = personality_keywords.get(personality, [])
        if not keywords:
            return chunks

        # Calculate personality relevance scores
        for chunk in chunks:
            content_lower = chunk.content.lower()
            personality_score = 0

            for keyword in keywords:
                if keyword in content_lower:
                    personality_score += content_lower.count(keyword) * 0.2

            # Store original score and add personality boost
            chunk.metadata['original_score'] = chunk.score
            chunk.score = chunk.score + personality_score

        return sorted(chunks, key=lambda x: x.score, reverse=True)


class MultiAgentRAGSystem:
    """Enhanced multi-agent RAG system with cross-perspective retrieval"""

    def __init__(self, agent_configs: Dict[str, Dict[str, Any]]):
        """Initialize multi-agent RAG system with enhanced retrievers"""
        self.agent_configs = agent_configs
        self.retrievers = {}

        # Initialize enhanced retrievers for each agent
        for agent_name, config in agent_configs.items():
            logger.info(f"Initializing enhanced retriever for {agent_name}...")
            retriever = EnhancedQdrantRAGRetriever(
                qdrant_host=config.get("host", "localhost"),
                qdrant_port=config.get("port", 6333),
                collection_name=config.get("collection", f"{agent_name.lower()}_chunks"),
                embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
                api_key=config.get("api_key")
            )
            self.retrievers[agent_name] = retriever

    def search_for_agent(
            self,
            agent_name: str,
            query: str,
            personality: str = "dove",
            negotiation_phase: str = "information_exchange",
            **search_kwargs
    ) -> RAGContext:
        """Enhanced search for specific agent with personality and phase awareness"""
        if agent_name not in self.retrievers:
            raise ValueError(f"Agent '{agent_name}' not found. Available agents: {list(self.retrievers.keys())}")

        retriever = self.retrievers[agent_name]

        personality_keywords = self._get_personality_keywords(personality)
        phase_keywords = self._get_phase_keywords(negotiation_phase)

        retrieved_chunks = retriever.retrieve_with_keywords(
            query=query,
            personality_keywords=personality_keywords,
            phase_keywords=phase_keywords,
            **search_kwargs
        )

        context = retriever.build_adaptive_context(
            query=query,
            retrieved_chunks=retrieved_chunks,
            agent_personality=personality,
            negotiation_phase=negotiation_phase
        )

        return context

    def retrieve_multiple_perspectives(
            self,
            query: str,
            primary_agent: str,
            personality: str = "dove",
            negotiation_phase: str = "information_exchange",
            include_opposing: bool = True,
            **search_kwargs
    ) -> Dict[str, RAGContext]:
        """Retrieve multiple perspectives on a topic"""
        results = {}

        results["primary"] = self.search_for_agent(
            primary_agent, query, personality, negotiation_phase, **search_kwargs
        )

        if include_opposing:
            opposing_agent = self._get_opposing_agent(primary_agent)
            if opposing_agent:
                try:
                    results["opposing"] = self.search_for_agent(
                        opposing_agent, query, personality, negotiation_phase,
                        top_k=search_kwargs.get("top_k", 3) // 2  # Fewer opposing chunks
                    )
                except Exception as e:
                    logger.warning(f"Could not retrieve opposing perspective: {e}")

        return results

    def _get_personality_keywords(self, personality: str) -> List[str]:
        """Get keywords associated with personality type"""
        personality_keywords = {
            "hawk": ["security", "threat", "defense", "strength", "deterrence"],
            "dove": ["cooperation", "peace", "dialogue", "compromise", "partnership"],
            "economist": ["economic", "trade", "market", "investment", "profit"],
            "legalist": ["law", "treaty", "legal", "binding", "international"],
            "innovator": ["innovation", "technology", "creative", "experimental"]
        }
        return personality_keywords.get(personality, [])

    def _get_phase_keywords(self, phase: str) -> List[str]:
        """Get keywords associated with negotiation phase"""
        phase_keywords = {
            "opening": ["position", "priority", "objective", "goal"],
            "information_exchange": ["background", "context", "situation", "facts"],
            "bargaining": ["offer", "proposal", "exchange", "concession"],
            "problem_solving": ["solution", "alternative", "creative", "innovative"],
            "decision_making": ["agreement", "commitment", "decision", "final"],
            "closing": ["summary", "next steps", "implementation", "follow-up"]
        }
        return phase_keywords.get(phase, [])

    def _get_opposing_agent(self, agent_name: str) -> Optional[str]:
        """Get the opposing agent name"""
        agent_map = {
            "Agent_USA": "Agent_Russia",
            "Agent_Russia": "Agent_USA"
        }
        return agent_map.get(agent_name)

    def analyze_information_gaps(
            self,
            query: str,
            agent1_name: str,
            agent2_name: str,
            **search_kwargs
    ) -> Dict[str, Any]:
        """Analyze information asymmetries between agents"""

        # Get contexts for both agents
        agent1_context = self.search_for_agent(agent1_name, query, **search_kwargs)
        agent2_context = self.search_for_agent(agent2_name, query, **search_kwargs)

        # Analyze differences
        analysis = {
            "query": query,
            "agent1_sources": set(agent1_context.sources),
            "agent2_sources": set(agent2_context.sources),
            "shared_sources": set(agent1_context.sources) & set(agent2_context.sources),
            "unique_to_agent1": set(agent1_context.sources) - set(agent2_context.sources),
            "unique_to_agent2": set(agent2_context.sources) - set(agent1_context.sources),
            "information_overlap": len(set(agent1_context.sources) & set(agent2_context.sources)) /
                                   max(1, len(set(agent1_context.sources) | set(agent2_context.sources))),
            "agent1_chunk_count": len(agent1_context.retrieved_chunks),
            "agent2_chunk_count": len(agent2_context.retrieved_chunks)
        }

        return analysis


def create_rag_system(agent_configs: Dict[str, Dict[str, Any]]) -> MultiAgentRAGSystem:
    """Factory function to create the RAG system"""
    try:
        return MultiAgentRAGSystem(agent_configs)
    except Exception as e:
        logger.error(f"Failed to create RAG system: {e}")
        raise