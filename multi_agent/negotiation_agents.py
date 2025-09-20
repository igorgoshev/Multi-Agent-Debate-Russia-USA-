#!/usr/bin/env python3
"""
Enhanced AI Agent Negotiation System
Improved negotiation agents with enhanced RAG integration and configuration support

Requirements:
pip install requests sentence-transformers qdrant-client
"""

import asyncio
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import uuid

# Import enhanced RAG system
from rag_system import MultiAgentRAGSystem, RAGContext, create_rag_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NegotiationPhase(Enum):
    """Phases of negotiation"""
    OPENING = "opening"
    INFORMATION_EXCHANGE = "information_exchange"
    BARGAINING = "bargaining"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"
    CLOSING = "closing"


class MessageType(Enum):
    """Types of negotiation messages"""
    PROPOSAL = "proposal"
    COUNTER_PROPOSAL = "counter_proposal"
    QUESTION = "question"
    INFORMATION = "information"
    CONCESSION = "concession"
    REJECTION = "rejection"
    ACCEPTANCE = "acceptance"
    CLARIFICATION = "clarification"


@dataclass
class NegotiationMessage:
    """A message in the negotiation"""
    id: str
    from_agent: str
    to_agent: str
    message_type: MessageType
    content: str
    phase: NegotiationPhase
    timestamp: str
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)
    rag_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if isinstance(self.message_type, MessageType):
            d["message_type"] = self.message_type.value
        if isinstance(self.phase, NegotiationPhase):
            d["phase"] = self.phase.value
        return d


@dataclass
class NegotiationState:
    """Current state of the negotiation"""
    negotiation_id: str
    participants: List[str]
    current_phase: NegotiationPhase
    topic: str
    messages: List[NegotiationMessage] = field(default_factory=list)
    positions: Dict[str, Any] = field(default_factory=dict)
    agreements: List[str] = field(default_factory=list)
    disagreements: List[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"


class OllamaClient:
    """Optimized client for Ollama API"""

    def __init__(self, host: str = "localhost", port: int = 11434):
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/api"

    def is_available(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = []

                if isinstance(data, dict) and "models" in data:
                    for model in data["models"]:
                        if isinstance(model, dict) and "name" in model:
                            models.append(model["name"])

                return models
            return []
        except Exception:
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull a model if not available"""
        try:
            response = requests.post(
                f"{self.api_url}/pull",
                json={"name": model_name},
                timeout=300
            )
            return response.status_code == 200
        except Exception:
            return False

    async def generate_response(
            self,
            model: str,
            messages: List[Dict[str, str]],
            system_prompt: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 1000
    ) -> str:
        """Generate response using Ollama API"""
        try:
            formatted_messages = []
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            formatted_messages.extend(messages)

            payload = {
                "model": model,
                "messages": formatted_messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict):
                    msg = result.get("message")
                    if isinstance(msg, dict) and "content" in msg:
                        return msg["content"]
                return str(result)
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "I apologize, but I'm having trouble processing your request."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your request."


class EnhancedNegotiationAgent:
    """Enhanced negotiation agent with adaptive personality and improved RAG integration"""

    def __init__(
            self,
            agent_name: str,
            agent_description: str,
            personality: str,
            rag_system: MultiAgentRAGSystem,
            model: str = "qwen3:8b",
            ollama_client: Optional[OllamaClient] = None,
            temperature: float = 0.7,
            max_tokens: int = 1000
    ):
        """
        Initialize enhanced negotiation agent

        Args:
            agent_name: Name of the agent (e.g., Agent_USA)
            agent_description: Description of agent's role and perspective
            personality: Agent personality type (hawk, dove, economist, etc.)
            rag_system: Enhanced multi-agent RAG system
            model: Ollama model to use
            ollama_client: Ollama client instance
            temperature: Model creativity parameter
            max_tokens: Maximum response tokens
        """
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.personality = personality
        self.rag_system = rag_system
        self.model = model
        self.ollama_client = ollama_client or OllamaClient()
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._ensure_model_available()

        self.system_prompt = self._create_enhanced_system_prompt()

        logger.info(f"Initialized {personality} negotiation agent {agent_name} with model {model}")

    def _ensure_model_available(self):
        """Ensure the model is available in Ollama"""
        available_models = self.ollama_client.list_models()
        if not any(self.model in m for m in available_models):
            logger.warning(f"Model {self.model} not found. Attempting to pull...")
            if not self.ollama_client.pull_model(self.model):
                fallback_models = ["llama3:8b", "qwen3:8b", "mistral:7b"]
                for fallback in fallback_models:
                    if any(fallback in m for m in available_models):
                        logger.info(f"Using fallback model {fallback}")
                        self.model = fallback
                        return
                raise ValueError(f"No suitable models available. Please install a supported model.")

    def _create_enhanced_system_prompt(self) -> str:
        """Create enhanced system prompt with personality integration"""
        personality_traits = self._get_personality_traits()

        return (
            f"You are {self.agent_name}, an AI negotiation agent with {self.personality} personality.\n\n"
            f"{self.agent_description}\n\n"
            f"PERSONALITY TRAITS ({self.personality.upper()}):\n"
            f"{personality_traits}\n\n"
            "CORE PRINCIPLES:\n"
            "1. Use your knowledge base to support arguments with factual information\n"
            "2. Maintain consistency with your personality and perspective\n"
            "3. Adapt your strategy based on negotiation phase and opponent behavior\n"
            "4. Seek outcomes that align with your agent's interests\n"
            "5. Be strategic but maintain diplomatic professionalism\n\n"
            "RESPONSE FORMAT:\n"
            "Respond with valid JSON containing these fields:\n"
            '{\n'
            '    "message_type": "proposal|counter_proposal|question|information|concession|rejection|acceptance|clarification",\n'
            '    "content": "Your negotiation message",\n'
            '    "confidence": 0.8,\n'
            '    "reasoning": "Strategic reasoning for this response",\n'
            '    "key_points": ["Main argument 1", "Main argument 2"],\n'
            '    "information_requests": ["What you need to know"]\n'
            '}\n\n'
            "Always maintain your personality perspective while being diplomatic and professional."
        )

    def _get_personality_traits(self) -> str:
        """Get personality-specific traits and behaviors"""
        traits = {
            "hawk": (
                "- Take strong positions and defend them vigorously\n"
                "- Prioritize security and national interests above compromise\n"
                "- Use historical precedents and military deterrence in arguments\n"
                "- Demand significant concessions before offering any\n"
                "- Frame negotiations in terms of strength and security"
            ),
            "dove": (
                "- Seek collaborative solutions and mutual benefits\n"
                "- Emphasize shared interests and long-term relationships\n"
                "- Offer incremental concessions to build negotiation momentum\n"
                "- Use inclusive language and acknowledge valid concerns\n"
                "- Focus on win-win outcomes and peaceful resolution"
            ),
            "economist": (
                "- Focus on quantifiable economic benefits and trade opportunities\n"
                "- Use data, statistics, and market analysis in arguments\n"
                "- Willing to compromise on political issues for economic gains\n"
                "- Propose phased implementation with economic incentives\n"
                "- Frame agreements in terms of competitiveness and growth"
            ),
            "legalist": (
                "- Reference international law, treaties, and legal precedents\n"
                "- Insist on legally binding agreements with enforcement mechanisms\n"
                "- Use formal diplomatic language and established procedures\n"
                "- Quote specific legal frameworks and court decisions\n"
                "- Emphasize compliance and institutional legitimacy"
            ),
            "innovator": (
                "- Propose creative solutions and experimental approaches\n"
                "- Focus on technology and innovation as solution enablers\n"
                "- Embrace risk-taking for potentially transformative outcomes\n"
                "- Suggest pilot programs and adaptive implementation\n"
                "- Challenge traditional approaches with new frameworks"
            )
        }
        return traits.get(self.personality, traits["dove"])

    async def search_enhanced_knowledge_base(
            self,
            query: str,
            negotiation_phase: str,
            include_opposing: bool = False,
            top_k: int = 3
    ) -> Tuple[str, List[str]]:
        """Enhanced knowledge base search with personality and phase awareness"""
        try:
            if include_opposing:
                # Get multiple perspectives
                perspectives = self.rag_system.retrieve_multiple_perspectives(
                    query=query,
                    primary_agent=self.agent_name,
                    personality=self.personality,
                    negotiation_phase=negotiation_phase,
                    top_k=top_k
                )

                context_info = f"SEARCH RESULTS FOR: '{query}'\n\n"
                sources = []

                # Primary perspective
                if "primary" in perspectives:
                    primary_context = perspectives["primary"]
                    context_info += f"YOUR PERSPECTIVE ({len(primary_context.retrieved_chunks)} sources):\n"
                    for i, chunk in enumerate(primary_context.retrieved_chunks, 1):
                        context_info += f"{i}. [Score: {chunk.score:.3f}] {chunk.content[:200]}...\n\n"
                    sources.extend(primary_context.sources)

                # Opposing perspective
                if "opposing" in perspectives:
                    opposing_context = perspectives["opposing"]
                    context_info += f"OPPOSING PERSPECTIVE ({len(opposing_context.retrieved_chunks)} sources):\n"
                    for i, chunk in enumerate(opposing_context.retrieved_chunks, 1):
                        context_info += f"{i}. [Score: {chunk.score:.3f}] {chunk.content[:150]}...\n\n"
                    sources.extend(opposing_context.sources)

                return context_info, list(set(sources))

            else:
                rag_context = self.rag_system.search_for_agent(
                    agent_name=self.agent_name,
                    query=query,
                    personality=self.personality,
                    negotiation_phase=negotiation_phase,
                    top_k=top_k
                )

                if not rag_context.retrieved_chunks:
                    return f"No relevant information found for: '{query}'", []

                context_info = f"KNOWLEDGE BASE RESULTS FOR: '{query}'\n\n"
                for i, chunk in enumerate(rag_context.retrieved_chunks, 1):
                    source = chunk.metadata.get("source", "Unknown")
                    context_info += f"{i}. [Score: {chunk.score:.3f}] [Source: {source}]\n"
                    context_info += f"{chunk.content[:250]}...\n\n"

                return context_info, rag_context.sources

        except Exception as e:
            logger.error(f"Enhanced RAG retrieval error: {e}")
            return f"Error retrieving information: {str(e)}", []

    async def process_message(
            self,
            message: str,
            opponent_name: str,
            negotiation_state: NegotiationState,
            context_queries: Optional[List[str]] = None,
            include_opposing_views: bool = False
    ) -> Dict[str, Any]:
        """
        Enhanced message processing with adaptive context retrieval

        Args:
            message: Message from opponent
            opponent_name: Name of opponent agent
            negotiation_state: Current negotiation state
            context_queries: Queries for context retrieval
            include_opposing_views: Whether to include opposing perspectives

        Returns:
            Structured negotiation response with enhanced context
        """
        # Enhanced context retrieval
        context_information = ""
        all_sources = []

        if context_queries:
            for query in context_queries:
                try:
                    context_info, sources = await self.search_enhanced_knowledge_base(
                        query=query,
                        negotiation_phase=negotiation_state.current_phase.value,
                        include_opposing=include_opposing_views,
                        top_k=3
                    )
                    context_information += f"\n{context_info}\n{'-' * 50}\n"
                    all_sources.extend(sources)
                except Exception as e:
                    logger.warning(f"Context retrieval failed for '{query}': {e}")

        recent_messages = negotiation_state.messages[-4:] if negotiation_state.messages else []
        history_text = "\n".join([
            f"{msg.from_agent} [{msg.message_type.value}]: {msg.content}"
            for msg in recent_messages
        ])

        user_message = (
            f"NEGOTIATION CONTEXT:\n"
            f"Phase: {negotiation_state.current_phase.value}\n"
            f"Your personality: {self.personality}\n"
            f"Opponent: {opponent_name}\n\n"
            f"OPPONENT'S MESSAGE:\n{message}\n\n"
            f"RECENT CONVERSATION:\n{history_text}\n\n"
            f"RELEVANT KNOWLEDGE:\n{context_information}\n\n"
            "Respond strategically based on your personality, the negotiation phase, and available information. "
            "Use the knowledge base information to support your arguments where relevant."
        )

        try:
            messages = [{"role": "user", "content": user_message}]

            response_text = await self.ollama_client.generate_response(
                model=self.model,
                messages=messages,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            try:
                response_data = json.loads(response_text)

                required_fields = ["message_type", "content", "confidence", "reasoning"]
                for field in required_fields:
                    if field not in response_data:
                        raise ValueError(f"Missing required field: {field}")

                response_data.setdefault("key_points", [])
                response_data.setdefault("information_requests", [])

                confidence = float(response_data.get("confidence", 0.5))
                response_data["confidence"] = max(0.0, min(1.0, confidence))

                response_data["rag_sources"] = list(set(all_sources))

                return response_data

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON parsing failed: {e}")
                fallback_content = self._generate_personality_fallback(message)

                return {
                    "message_type": MessageType.CLARIFICATION.value,
                    "content": fallback_content,
                    "confidence": 0.4,
                    "reasoning": f"JSON parsing failed, using personality-based fallback",
                    "key_points": [f"Maintaining {self.personality} perspective"],
                    "information_requests": ["Could you clarify your main points?"],
                    "rag_sources": all_sources
                }

        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            return {
                "message_type": MessageType.CLARIFICATION.value,
                "content": "I need to carefully consider this matter. Could you provide more details?",
                "confidence": 0.3,
                "reasoning": f"Processing error: {str(e)}",
                "key_points": ["Need more information"],
                "information_requests": ["Please clarify your position"],
                "rag_sources": []
            }

    def _generate_personality_fallback(self, message: str) -> str:
        """Generate personality-appropriate fallback response"""
        fallbacks = {
            "hawk": "I need to carefully assess the security implications of your proposal before responding.",
            "dove": "Thank you for sharing your perspective. I believe we can find common ground here.",
            "economist": "Let me consider the economic implications and potential mutual benefits of this approach.",
            "legalist": "I need to review the legal frameworks and precedents relevant to this matter.",
            "innovator": "This presents an interesting challenge that might benefit from a creative approach."
        }
        return fallbacks.get(self.personality, "I need to consider this matter more carefully.")


class EnhancedNegotiationManager:
    """Enhanced negotiation manager with improved phase management and analysis"""

    def __init__(
            self,
            agent1: EnhancedNegotiationAgent,
            agent2: EnhancedNegotiationAgent,
            topic: str,
            max_rounds: int = 8,
            phase_transition_threshold: int = 2
    ):
        self.agent1 = agent1
        self.agent2 = agent2
        self.topic = topic
        self.max_rounds = max_rounds
        self.phase_transition_threshold = phase_transition_threshold

        self.negotiation_state = NegotiationState(
            negotiation_id=str(uuid.uuid4()),
            participants=[agent1.agent_name, agent2.agent_name],
            current_phase=NegotiationPhase.OPENING,
            topic=topic
        )

        logger.info(
            f"Enhanced negotiation: {agent1.agent_name} ({agent1.personality}) vs {agent2.agent_name} ({agent2.personality})")

    def add_message(
            self,
            from_agent: str,
            to_agent: str,
            message_type: MessageType,
            content: str,
            confidence: float,
            supporting_evidence: Optional[List[str]] = None,
            rag_sources: Optional[List[str]] = None
    ):
        """Add enhanced message with source tracking"""
        message = NegotiationMessage(
            id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            phase=self.negotiation_state.current_phase,
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            supporting_evidence=supporting_evidence or [],
            rag_sources=rag_sources or []
        )

        self.negotiation_state.messages.append(message)
        self.negotiation_state.last_activity = datetime.now().isoformat()

    async def conduct_negotiation_round(
            self,
            initiating_agent: str,
            message: str,
            context_queries: Optional[Dict[str, List[str]]] = None,
            include_opposing_views: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Enhanced negotiation round with improved context handling"""

        if initiating_agent == self.agent1.agent_name:
            first_agent, second_agent = self.agent1, self.agent2
        else:
            first_agent, second_agent = self.agent2, self.agent1

        first_queries = context_queries.get(first_agent.agent_name, []) if context_queries else None
        first_response = await first_agent.process_message(
            message=message,
            opponent_name=second_agent.agent_name,
            negotiation_state=self.negotiation_state,
            context_queries=first_queries,
            include_opposing_views=include_opposing_views
        )

        self.add_message(
            from_agent=first_agent.agent_name,
            to_agent=second_agent.agent_name,
            message_type=MessageType(first_response.get("message_type", "information")),
            content=str(first_response.get("content", "")),
            confidence=float(first_response.get("confidence", 0.5)),
            supporting_evidence=first_response.get("key_points", []),
            rag_sources=first_response.get("rag_sources", [])
        )

        second_queries = context_queries.get(second_agent.agent_name, []) if context_queries else None
        second_response = await second_agent.process_message(
            message=str(first_response.get("content", "")),
            opponent_name=first_agent.agent_name,
            negotiation_state=self.negotiation_state,
            context_queries=second_queries,
            include_opposing_views=include_opposing_views
        )

        self.add_message(
            from_agent=second_agent.agent_name,
            to_agent=first_agent.agent_name,
            message_type=MessageType(second_response.get("message_type", "information")),
            content=str(second_response.get("content", "")),
            confidence=float(second_response.get("confidence", 0.5)),
            supporting_evidence=second_response.get("key_points", []),
            rag_sources=second_response.get("rag_sources", [])
        )

        return first_response, second_response

    async def run_full_negotiation(
            self,
            initial_prompt: str,
            include_opposing_views: bool = False
    ) -> Dict[str, Any]:
        """Run complete negotiation with enhanced features"""

        logger.info(f"Starting enhanced negotiation: {self.topic}")

        current_message = initial_prompt
        current_agent = self.agent1.agent_name

        for round_num in range(self.max_rounds):
            logger.info(
                f"=== Round {round_num + 1}/{self.max_rounds} - Phase: {self.negotiation_state.current_phase.value} ===")

            context_queries = self._generate_enhanced_context_queries()

            try:
                response1, response2 = await self.conduct_negotiation_round(
                    initiating_agent=current_agent,
                    message=current_message,
                    context_queries=context_queries,
                    include_opposing_views=include_opposing_views
                )

                self._log_round_results(response1, response2, round_num)

                if self._should_terminate_negotiation(response1, response2):
                    logger.info("Negotiation terminated based on agent responses")
                    break

                if (round_num + 1) % self.phase_transition_threshold == 0:
                    self._advance_negotiation_phase()

                current_message = str(response2.get("content", ""))
                current_agent = (self.agent2.agent_name
                                 if current_agent == self.agent1.agent_name
                                 else self.agent1.agent_name)

                await asyncio.sleep(0.5)  # Brief pause

            except Exception as e:
                logger.error(f"Error in round {round_num + 1}: {e}")
                break

        self.negotiation_state.status = "completed"
        return self._analyze_enhanced_results()

    def _generate_enhanced_context_queries(self) -> Dict[str, List[str]]:
        """Generate context queries based on personalities and phase"""
        base_queries = [self.topic]

        # Phase-specific queries
        phase_queries = {
            NegotiationPhase.OPENING: ["negotiation strategy", "initial positions"],
            NegotiationPhase.INFORMATION_EXCHANGE: ["background information", "key facts"],
            NegotiationPhase.BARGAINING: ["proposals", "trade-offs", "concessions"],
            NegotiationPhase.PROBLEM_SOLVING: ["creative solutions", "alternatives"],
            NegotiationPhase.DECISION_MAKING: ["final agreements", "implementation"],
            NegotiationPhase.CLOSING: ["summary", "next steps"]
        }

        phase_specific = phase_queries.get(self.negotiation_state.current_phase, [])

        personality_queries = {
            "hawk": ["security concerns", "military aspects", "deterrence"],
            "dove": ["cooperation opportunities", "mutual benefits", "peace"],
            "economist": ["economic impact", "trade benefits", "market analysis"],
            "legalist": ["legal framework", "international law", "treaties"],
            "innovator": ["innovative solutions", "technology", "pilot programs"]
        }

        agent1_queries = base_queries + phase_specific + personality_queries.get(self.agent1.personality, [])
        agent2_queries = base_queries + phase_specific + personality_queries.get(self.agent2.personality, [])

        return {
            self.agent1.agent_name: agent1_queries,
            self.agent2.agent_name: agent2_queries
        }

    def _log_round_results(self, response1: Dict[str, Any], response2: Dict[str, Any], round_num: int):
        """Enhanced logging of round results"""
        print(f"\n{'=' * 80}")
        print(f"ROUND {round_num + 1} - {self.negotiation_state.current_phase.value.upper()}")
        print(f"{'=' * 80}")

        print(f"\n{self.agent1.agent_name} ({self.agent1.personality}) [{response1.get('message_type', 'unknown')}]:")
        print(f"Confidence: {float(response1.get('confidence', 0)):.2f}")
        print(f"Content: {str(response1.get('content', ''))}")
        if response1.get('key_points'):
            print(f"Key Points: {', '.join(response1['key_points'])}")

        print(f"\n{self.agent2.agent_name} ({self.agent2.personality}) [{response2.get('message_type', 'unknown')}]:")
        print(f"Confidence: {float(response2.get('confidence', 0)):.2f}")
        print(f"Content: {str(response2.get('content', ''))}")
        if response2.get('key_points'):
            print(f"Key Points: {', '.join(response2['key_points'])}")

    def _should_terminate_negotiation(self, response1: Dict[str, Any], response2: Dict[str, Any]) -> bool:
        """Enhanced termination logic"""
        if (response1.get("message_type") == "acceptance" and
                response2.get("message_type") == "acceptance"):
            return True

        if (response1.get("message_type") == "rejection" and
                response2.get("message_type") == "rejection" and
                self.negotiation_state.current_phase in [NegotiationPhase.DECISION_MAKING, NegotiationPhase.CLOSING]):
            return True

        if (float(response1.get("confidence", 1.0)) < 0.25 and
                float(response2.get("confidence", 1.0)) < 0.25):
            return True

        return False

    def _advance_negotiation_phase(self):
        """Advance to next negotiation phase"""
        phases = list(NegotiationPhase)
        try:
            current_index = phases.index(self.negotiation_state.current_phase)
            if current_index < len(phases) - 1:
                self.negotiation_state.current_phase = phases[current_index + 1]
                logger.info(f"Advanced to phase: {self.negotiation_state.current_phase.value}")
        except ValueError:
            pass

    def _analyze_enhanced_results(self) -> Dict[str, Any]:
        """Enhanced result analysis with personality and source tracking"""
        agreements = []
        disagreements = []
        proposals = []

        for message in self.negotiation_state.messages:
            if message.message_type == MessageType.ACCEPTANCE:
                agreements.append(message.content)
            elif message.message_type == MessageType.REJECTION:
                disagreements.append(message.content)
            elif message.message_type == MessageType.PROPOSAL:
                proposals.append(message.content)

        total_messages = len(self.negotiation_state.messages)
        avg_confidence = (sum(
            msg.confidence for msg in self.negotiation_state.messages) / total_messages) if total_messages > 0 else 0.0

        agent1_messages = [msg for msg in self.negotiation_state.messages if msg.from_agent == self.agent1.agent_name]
        agent2_messages = [msg for msg in self.negotiation_state.messages if msg.from_agent == self.agent2.agent_name]

        all_rag_sources = []
        for msg in self.negotiation_state.messages:
            all_rag_sources.extend(msg.rag_sources)
        unique_sources = list(set(all_rag_sources))

        personality_interaction = f"{self.agent1.personality}_vs_{self.agent2.personality}"

        return {
            "negotiation_id": self.negotiation_state.negotiation_id,
            "topic": self.topic,
            "participants": {
                self.agent1.agent_name: self.agent1.personality,
                self.agent2.agent_name: self.agent2.personality
            },
            "personality_interaction": personality_interaction,
            "duration": self._calculate_duration(),
            "total_rounds": total_messages // 2 if total_messages >= 2 else total_messages,
            "final_phase": self.negotiation_state.current_phase.value,
            "status": self.negotiation_state.status,
            "outcomes": {
                "agreements": agreements,
                "disagreements": disagreements,
                "proposals": proposals,
                "agreement_ratio": len(agreements) / max(1, len(agreements) + len(disagreements))
            },
            "metrics": {
                "total_messages": total_messages,
                "average_confidence": avg_confidence,
                "agent1_message_count": len(agent1_messages),
                "agent2_message_count": len(agent2_messages),
                "unique_rag_sources": len(unique_sources),
                "total_rag_retrievals": len(all_rag_sources)
            },
            "knowledge_sources": unique_sources,
            "message_history": [msg.to_dict() for msg in self.negotiation_state.messages]
        }

    def _calculate_duration(self) -> str:
        """Calculate negotiation duration"""
        try:
            start = datetime.fromisoformat(self.negotiation_state.started_at)
            end = datetime.fromisoformat(self.negotiation_state.last_activity)
            return str(end - start)
        except Exception:
            return "unknown"

    def save_negotiation_log(self, filename: str):
        """Save enhanced negotiation log"""
        results = self._analyze_enhanced_results()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Enhanced negotiation log saved to {filename}")


async def setup_enhanced_system(
        usa_model: str = "qwen3:8b",
        russia_model: str = "llama3:8b",
        usa_personality: str = "dove",
        russia_personality: str = "hawk",
        agent_configs: Optional[Dict[str, Dict[str, Any]]] = None
) -> Optional[Tuple[EnhancedNegotiationAgent, EnhancedNegotiationAgent, MultiAgentRAGSystem]]:
    """Setup enhanced negotiation system with configuration support"""

    print("🚀 Starting Enhanced AI Agent Negotiation System")
    print("=" * 60)

    ollama_client = OllamaClient()
    if not ollama_client.is_available():
        print("❌ Ollama server not running!")
        print("Please start Ollama: ollama serve")
        return None

    print("✅ Ollama server is running")

    if agent_configs is None:
        agent_configs = {
            "Agent_USA": {
                "host": "localhost",
                "port": 6333,
                "collection": "usa_collection",
                "embedding_model": "all-MiniLM-L6-v2"
            },
            "Agent_Russia": {
                "host": "localhost",
                "port": 6334,
                "collection": "russia_collection",
                "embedding_model": "all-MiniLM-L6-v2"
            }
        }

    try:
        print("🔄 Initializing enhanced RAG system...")
        rag_system = create_rag_system(agent_configs)

        print(f"🤖 Creating enhanced negotiation agents...")
        print(f"   USA Agent: {usa_personality} personality, {usa_model} model")
        print(f"   Russia Agent: {russia_personality} personality, {russia_model} model")

        agent_usa = EnhancedNegotiationAgent(
            agent_name="Agent_USA",
            agent_description=_get_agent_description("usa", usa_personality),
            personality=usa_personality,
            rag_system=rag_system,
            model=usa_model,
            ollama_client=ollama_client
        )

        agent_russia = EnhancedNegotiationAgent(
            agent_name="Agent_Russia",
            agent_description=_get_agent_description("russia", russia_personality),
            personality=russia_personality,
            rag_system=rag_system,
            model=russia_model,
            ollama_client=ollama_client
        )

        return agent_usa, agent_russia, rag_system

    except Exception as e:
        print(f"❌ Error setting up enhanced system: {e}")
        return None


def _get_agent_description(country: str, personality: str) -> str:

    base_descriptions = {
        "usa": {
            "core": "You represent the United States perspective in international negotiations.",
            "values": "democratic values, international law, NATO alliance strength, collective security"
        },
        "russia": {
            "core": "You represent the Russian perspective in international negotiations.",
            "values": "national security, sphere of influence, historical ties, regional stability"
        }
    }

    personality_adaptations = {
        "hawk": "Take strong security-focused positions and emphasize military deterrence.",
        "dove": "Seek diplomatic solutions and emphasize cooperation and mutual benefits.",
        "economist": "Focus on economic implications and trade relationships in all discussions.",
        "legalist": "Emphasize international law, treaties, and legal frameworks.",
        "innovator": "Propose creative solutions and technological approaches to problems."
    }

    country_info = base_descriptions.get(country, base_descriptions["usa"])
    personality_info = personality_adaptations.get(personality, personality_adaptations["dove"])

    return (
        f"{country_info['core']}\n\n"
        f"CORE VALUES: {country_info['values']}\n\n"
        f"PERSONALITY APPROACH: {personality_info}\n\n"
        f"Maintain consistency with both your national perspective and {personality} personality traits."
    )


async def run_configured_negotiation(config) -> Dict[str, Any]:
    """Run negotiation with provided configuration (integration point for config_negotiation.py)"""

    print(f"🎭 Running configured negotiation:")
    print(f"   Scenario: {config.scenario_name}")
    print(f"   USA: {config.usa_personality} ({config.usa_model})")
    print(f"   Russia: {config.russia_personality} ({config.russia_model})")
    print(f"   Rounds: {config.max_rounds}")

    system_result = await setup_enhanced_system(
        usa_model=config.usa_model,
        russia_model=config.russia_model,
        usa_personality=config.usa_personality,
        russia_personality=config.russia_personality
    )

    if not system_result:
        raise RuntimeError("Failed to setup negotiation system")

    agent_usa, agent_russia, rag_system = system_result

    manager = EnhancedNegotiationManager(
        agent1=agent_usa,
        agent2=agent_russia,
        topic=config.topic,
        max_rounds=config.max_rounds,
        phase_transition_threshold=config.phase_transition_threshold
    )

    results = await manager.run_full_negotiation(
        initial_prompt=config.initial_prompt,
        include_opposing_views=True  # Enhanced feature //TODO WATCH OUT
    )

    if config.save_detailed_logs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"negotiation_{config.scenario_name}_{config.usa_personality}v{config.russia_personality}_{timestamp}.json"
        manager.save_negotiation_log(filename)
        print(f"📄 Detailed log saved to {filename}")

    return results


async def main_enhanced():
    """Enhanced main function with default configuration"""

    setup_result = await setup_enhanced_system(
        usa_personality="dove",
        russia_personality="hawk"
    )

    if not setup_result:
        return

    agent_usa, agent_russia, rag_system = setup_result

    manager = EnhancedNegotiationManager(
        agent1=agent_usa,
        agent2=agent_russia,
        topic="Ukraine conflict resolution and future security arrangements",
        max_rounds=6
    )

    initial_prompt = (
        "We are here to discuss a potential framework for resolving the ongoing conflict in Ukraine "
        "and establishing future security arrangements in Eastern Europe.\n\n"
        "Both parties should present their key concerns, priorities, and potential areas for compromise.\n\n"
        "The goal is to find a mutually acceptable path forward that addresses legitimate security concerns "
        "while respecting international law and sovereignty principles.\n\n"
        "Please begin by stating your primary position and key requirements for any potential agreement."
    )

    print("🎯 Starting enhanced negotiation...")
    print(f"Topic: {manager.topic}")
    print(
        f"Participants: {agent_usa.agent_name} ({agent_usa.personality}) vs {agent_russia.agent_name} ({agent_russia.personality})")

    results = await manager.run_full_negotiation(
        initial_prompt=initial_prompt,
        include_opposing_views=True
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manager.save_negotiation_log(f"enhanced_negotiation_{timestamp}.json")

    print("\n📊 ENHANCED NEGOTIATION SUMMARY")
    print("=" * 50)
    print(f"Personality Interaction: {results.get('personality_interaction')}")
    print(f"Duration: {results.get('duration')}")
    print(f"Total rounds: {results.get('total_rounds')}")
    print(f"Final phase: {results.get('final_phase')}")
    print(f"RAG sources used: {results.get('metrics', {}).get('unique_rag_sources', 0)}")
    print(f"Average confidence: {results.get('metrics', {}).get('average_confidence', 0.0):.2f}")
    print(f"Agreement ratio: {results.get('outcomes', {}).get('agreement_ratio', 0.0):.2f}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--enhanced":
        asyncio.run(main_enhanced())
    else:
        print("Enhanced AI Agent Negotiation System")
        print("Usage: python improved_negotiation_agents.py --enhanced")
        print("Or use with config_negotiation.py for full configuration options")