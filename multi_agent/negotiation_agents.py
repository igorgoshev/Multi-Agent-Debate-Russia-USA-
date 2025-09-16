#!/usr/bin/env python3
# negotiation_agents.py
import asyncio
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import uuid
import time

# Import our RAG system (user must provide rag_system.py with MultiAgentRAGSystem)
from rag_system import MultiAgentRAGSystem, RAGContext  # noqa: F401

# Setup logging
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

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert enums to their values for JSON serialization
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
    positions: Dict[str, Any] = field(default_factory=dict)  # Each agent's current position
    agreements: List[str] = field(default_factory=list)  # Points of agreement
    disagreements: List[str] = field(default_factory=list)  # Points of disagreement
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"  # active, completed, stalled


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, host: str = "localhost", port: int = 11434):
        self.base_url = f"http://{host}:{port}/"
        self.api_url = f"{self.base_url}api"

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
                # Try common shapes for the response
                models = []
                if isinstance(data, dict):
                    # Some Ollama endpoints return {"models": [{"name": "..."}]}
                    if "models" in data and isinstance(data["models"], list):
                        for m in data["models"]:
                            if isinstance(m, dict) and "name" in m:
                                models.append(m["name"])
                            elif isinstance(m, str):
                                models.append(m)
                    # Sometimes "tags" key may exist
                    elif "tags" in data and isinstance(data["tags"], list):
                        for t in data["tags"]:
                            if isinstance(t, dict) and "name" in t:
                                models.append(t["name"])
                            elif isinstance(t, str):
                                models.append(t)
                    else:
                        # If the dict looks like a mapping of names, just flatten strings
                        for v in data.values():
                            if isinstance(v, list):
                                for item in v:
                                    if isinstance(item, str):
                                        models.append(item)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "name" in item:
                            models.append(item["name"])
                        elif isinstance(item, str):
                            models.append(item)
                return models
            return []
        except Exception:
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull a model if not available"""
        try:
            print(f"🔄 Pulling model {model_name}...")
            response = requests.post(
                f"{self.api_url}/pull",
                json={"name": model_name},
                stream=True,
                timeout=300
            )
            success = False
            if response is None:
                return False
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    # Not JSON — print raw
                    print(line)
                    continue
                if "status" in data:
                    print(f"   {data['status']}")
                if data.get("status") == "success":
                    print(f"✅ Model {model_name} pulled successfully")
                    success = True
                    break
            return success or response.status_code == 200
        except Exception as e:
            print(f"❌ Failed to pull model {model_name}: {e}")
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
            # Format messages for Ollama
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
                # Try several possible shapes
                if isinstance(result, dict):
                    # Common shape: {"message": {"content": "..."}}
                    msg = result.get("message")
                    if isinstance(msg, dict) and "content" in msg:
                        return msg["content"]
                    # Sometimes result has "choices" list
                    choices = result.get("choices")
                    if isinstance(choices, list) and choices:
                        first = choices[0]
                        if isinstance(first, dict):
                            if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                                return first["message"]["content"]
                            if "text" in first:
                                return first["text"]
                    # Fallback to raw string of result
                    return json.dumps(result)
                else:
                    return str(result)
            else:
                logger.error(f"Ollama API error {response.status_code} - {response.text}")
                return "I apologize, but I'm having trouble processing your request right now."
        except Exception as e:
            logger.error(f"Error generating response {e}")
            return "I apologize, but I encountered an error while processing your request."


class RAGEnhancedNegotiationAgent:
    """Negotiation agent with RAG-enhanced knowledge retrieval using free models"""

    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        rag_system: MultiAgentRAGSystem,
        model: str = "qwen38b",
        ollama_client: Optional[OllamaClient] = None
    ):
        """
        Initialize negotiation agent

        Args:
            agent_name: Name of the agent (e.g., Agent_USA)
            agent_description: Description of agent's role and perspective
            rag_system: Multi-agent RAG system for knowledge retrieval
            model: Ollama model to use (qwen38b, llama3.38b, mistral7b)
            ollama_client: Ollama client instance
        """
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.rag_system = rag_system
        self.model = model
        self.ollama_client = ollama_client or OllamaClient()

        # Ensure model is available
        self._ensure_model_available()

        # Create system prompt
        self.system_prompt = self._create_system_prompt()

        logger.info(f"Initialized negotiation agent {agent_name} with model {model}")

    def _ensure_model_available(self):
        """Ensure the model is available in Ollama"""
        available_models = self.ollama_client.list_models()
        if not any(self.model in m for m in available_models):
            print(f"⚠️ Model {self.model} not found. Attempting to pull...")
            success = self.ollama_client.pull_model(self.model)
            if not success:
                # Fallback to basic model order
                fallback_models = ["llama3:8b", "qwen3:8b", "mistral7b"]
                for fallback in fallback_models:
                    if any(fallback in m for m in available_models):
                        print(f"🔄 Using fallback model {fallback}")
                        self.model = fallback
                        return
                raise ValueError(f"No suitable models available. Please run ollama pull {self.model}")

    def _create_system_prompt(self) -> str:
        """Create system prompt for the agent"""
        return (
            f"You are {self.agent_name}, an AI negotiation agent with the following characteristics\n\n"
            f"{self.agent_description}\n\n"
            "CORE PRINCIPLES\n"
            "1. You have access to a knowledge base with relevant information\n"
            "2. Always ground your arguments in factual information from your knowledge base\n"
            "3. Be strategic but ethical in your negotiation approach\n"
            "4. Maintain consistency with your agent's perspective and interests\n"
            "5. Seek win-win solutions when possible, but protect your core interests\n\n"
            "NEGOTIATION GUIDELINES\n"
            "- Start with information gathering and position establishment\n"
            "- Use evidence from your knowledge base to support positions\n"
            "- Ask clarifying questions when needed\n"
            "- Make proposals and counter-proposals based on available information\n"
            "- Be willing to make reasonable concessions for mutual benefit\n"
            "- Clearly state your reasoning and confidence level\n\n"
            "RESPONSE FORMAT REQUIREMENTS\n"
            "You must format your response as valid JSON with these exact fields\n"
            '{\n'
            '    "message_type": "proposal|counter_proposal|question|information|concession|rejection|acceptance|clarification",\n'
            '    "content": "Your actual negotiation message here",\n'
            '    "confidence": 0.8,\n'
            '    "reasoning": "Why you\'re taking this position/action",\n'
            '    "key_points": ["Main point 1", "Main point 2"],\n'
            '    "information_requests": ["Any information you need from the other party"]\n'
            '}\n\n'
            "IMPORTANT:\n"
            "- Always respond with valid JSON format\n"
            "- Confidence should be between 0.0 and 1.0\n"
            "- Be diplomatic but firm in representing {self.agent_name} interests\n"
            "- Use concrete examples and evidence when making arguments\n"
        )

    async def search_knowledge_base(
        self,
        query: str,
        top_k: int = 3
    ) -> str:
        """Search the agent's knowledge base for relevant information"""
        try:
            # Get RAG context for this agent
            rag_context: RAGContext = self.rag_system.search_for_agent(
                agent_name=self.agent_name,
                query=query,
                top_k=top_k,
                score_threshold=0.6
            )

            if not getattr(rag_context, "retrieved_chunks", None):
                return f"No relevant information found for query '{query}'"

            # Format the retrieved information
            chunks = rag_context.retrieved_chunks
            context_info = f"Retrieved {len(chunks)} relevant pieces of information\n\n"
            for i, chunk in enumerate(chunks, 1):
                score = getattr(chunk, "score", None)
                try:
                    score_text = f"{score:.3f}" if score is not None else "N/A"
                except Exception:
                    score_text = str(score)
                source = chunk.metadata.get("source", "Unknown") if getattr(chunk, "metadata", None) else "Unknown"
                content_snippet = (chunk.content[:200] + "...") if getattr(chunk, "content", None) else ""
                context_info += f"{i}. [Score {score_text}] [Source {source}]\n{content_snippet}\n\n"

            return context_info

        except Exception as e:
            logger.error(f"RAG retrieval error {e}")
            return f"Error retrieving information {str(e)}"

    async def process_message(
        self,
        message: str,
        opponent_name: str,
        negotiation_state: NegotiationState,
        context_queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process incoming message and generate response

        Args:
            message: Message from the opponent
            opponent_name: Name of the opponent agent
            negotiation_state: Current negotiation state
            context_queries: Optional queries to retrieve relevant context

        Returns:
            Structured negotiation response
        """
        # Retrieve relevant context if queries provided
        context_information = ""
        if context_queries:
            for query in context_queries:
                try:
                    context_info = await self.search_knowledge_base(query, top_k=2)
                    context_information += f"Relevant context for '{query}':\n{context_info}\n"
                except Exception as e:
                    logger.warning(f"Context retrieval failed for query '{query}': {e}")

        # Prepare conversation history (last 3 messages)
        recent_messages = negotiation_state.messages[-3:] if negotiation_state.messages else []
        history_text = "\n".join([
            f"{msg.from_agent}: {msg.content}" for msg in recent_messages
        ])

        # Prepare the full prompt
        user_message = (
            f"OPPONENT'S MESSAGE: {message}\n\n"
            f"CURRENT NEGOTIATION PHASE: {negotiation_state.current_phase.value}\n\n"
            f"RECENT CONVERSATION HISTORY:\n{history_text}\n\n"
            f"RELEVANT CONTEXT FROM YOUR KNOWLEDGE BASE:\n{context_information}\n\n"
            "Please respond with your negotiation strategy and message. Remember to format your response as valid JSON "
            "with the required fields: message_type, content, confidence, reasoning, key_points, and information_requests."
        )

        # Generate response using Ollama
        try:
            messages = [{"role": "user", "content": user_message}]

            response_text = await self.ollama_client.generate_response(
                model=self.model,
                messages=messages,
                system_prompt=self.system_prompt,
                temperature=0.7,
                max_tokens=1000
            )

            # Try to parse JSON response
            try:
                response_data = json.loads(response_text)
                # Validate required fields
                required_fields = ["message_type", "content", "confidence", "reasoning"]
                for field_name in required_fields:
                    if field_name not in response_data:
                        raise ValueError(f"Missing required field {field_name}")

                # Ensure default values for optional fields
                response_data.setdefault("key_points", [])
                response_data.setdefault("information_requests", [])

                # Validate confidence is in valid range
                try:
                    confidence = float(response_data.get("confidence", 0.5))
                except Exception:
                    confidence = 0.5
                if not (0.0 <= confidence <= 1.0):
                    response_data["confidence"] = max(0.0, min(1.0, confidence))
                else:
                    response_data["confidence"] = confidence

                return response_data

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                # Fallback response
                return {
                    "message_type": MessageType.INFORMATION.value,
                    "content": response_text,  # Use raw text if JSON parsing fails
                    "confidence": 0.5,
                    "reasoning": "Generated response but failed to parse structured format.",
                    "key_points": [],
                    "information_requests": []
                }

        except Exception as e:
            logger.error(f"Agent processing error {e}")
            # Fallback response
            return {
                "message_type": MessageType.CLARIFICATION.value,
                "content": "I need to consider this carefully. Could you provide more details about your position?",
                "confidence": 0.5,
                "reasoning": f"Processing error occurred: {str(e)}",
                "key_points": ["Need more information to respond appropriately"],
                "information_requests": ["Could you clarify your main concerns?"]
            }


class NegotiationManager:
    """Manages the negotiation between two agents"""

    def __init__(
        self,
        agent1: RAGEnhancedNegotiationAgent,
        agent2: RAGEnhancedNegotiationAgent,
        topic: str
    ):
        """
        Initialize negotiation manager

        Args:
            agent1: First negotiation agent
            agent2: Second negotiation agent
            topic: Topic of negotiation
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.topic = topic

        # Initialize negotiation state
        self.negotiation_state = NegotiationState(
            negotiation_id=str(uuid.uuid4()),
            participants=[agent1.agent_name, agent2.agent_name],
            current_phase=NegotiationPhase.OPENING,
            topic=topic,
            messages=[],
            positions={},
            agreements=[],
            disagreements=[],
            started_at=datetime.now().isoformat(),
            last_activity=datetime.now().isoformat(),
            status="active"
        )

        logger.info(f"Initialized negotiation {agent1.agent_name} vs {agent2.agent_name} on '{topic}'")

    def add_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: MessageType,
        content: str,
        confidence: float,
        supporting_evidence: Optional[List[str]] = None
    ):
        """Add a message to the negotiation history"""
        message = NegotiationMessage(
            id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            phase=self.negotiation_state.current_phase,
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            supporting_evidence=supporting_evidence or []
        )

        self.negotiation_state.messages.append(message)
        self.negotiation_state.last_activity = datetime.now().isoformat()

    async def conduct_negotiation_round(
        self,
        initiating_agent: str,
        message: str,
        context_queries: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Conduct one round of negotiation

        Args:
            initiating_agent: Which agent starts this round
            message: Initial message for the round
            context_queries: Queries for context retrieval per agent

        Returns:
            Tuple of responses from both agents
        """
        # Determine which agent goes first
        if initiating_agent == self.agent1.agent_name:
            first_agent, second_agent = self.agent1, self.agent2
        else:
            first_agent, second_agent = self.agent2, self.agent1

        # First agent processes the initial message/prompt
        first_queries = context_queries.get(first_agent.agent_name, []) if context_queries else None
        first_response = await first_agent.process_message(
            message=message,
            opponent_name=second_agent.agent_name,
            negotiation_state=self.negotiation_state,
            context_queries=first_queries
        )

        # Add first agent's response to history
        try:
            mt_value = first_response.get("message_type", MessageType.INFORMATION.value)
            msg_type_enum = MessageType(mt_value) if isinstance(mt_value, str) else mt_value
        except Exception:
            msg_type_enum = MessageType.INFORMATION
        self.add_message(
            from_agent=first_agent.agent_name,
            to_agent=second_agent.agent_name,
            message_type=msg_type_enum,
            content=str(first_response.get("content", "")),
            confidence=float(first_response.get("confidence", 0.5)),
            supporting_evidence=first_response.get("key_points", [])
        )

        # Second agent responds
        second_queries = context_queries.get(second_agent.agent_name, []) if context_queries else None
        second_response = await second_agent.process_message(
            message=str(first_response.get("content", "")),
            opponent_name=first_agent.agent_name,
            negotiation_state=self.negotiation_state,
            context_queries=second_queries
        )

        # Add second agent's response to history
        try:
            mt_value2 = second_response.get("message_type", MessageType.INFORMATION.value)
            msg_type_enum2 = MessageType(mt_value2) if isinstance(mt_value2, str) else mt_value2
        except Exception:
            msg_type_enum2 = MessageType.INFORMATION
        self.add_message(
            from_agent=second_agent.agent_name,
            to_agent=first_agent.agent_name,
            message_type=msg_type_enum2,
            content=str(second_response.get("content", "")),
            confidence=float(second_response.get("confidence", 0.5)),
            supporting_evidence=second_response.get("key_points", [])
        )

        return first_response, second_response

    async def run_full_negotiation(
        self,
        initial_prompt: str,
        max_rounds: int = 8,
        phase_transition_threshold: int = 2
    ) -> Dict[str, Any]:
        """
        Run a complete negotiation session

        Args:
            initial_prompt: Starting prompt for the negotiation
            max_rounds: Maximum number of negotiation rounds
            phase_transition_threshold: Rounds before considering phase transition

        Returns:
            Complete negotiation results
        """
        logger.info(f"Starting full negotiation {self.topic}")

        current_message = initial_prompt
        current_agent = self.agent1.agent_name

        for round_num in range(max_rounds):
            logger.info(f"=== Round {round_num + 1} ===")

            # Determine context queries based on negotiation phase and topic
            context_queries = self._generate_context_queries()

            try:
                # Conduct negotiation round
                response1, response2 = await self.conduct_negotiation_round(
                    initiating_agent=current_agent,
                    message=current_message,
                    context_queries=context_queries
                )

                # Log the responses (safe slicing)
                logger.info(f"{self.agent1.agent_name}: {str(response1.get('content', ''))[:100]}...")
                logger.info(f"{self.agent2.agent_name}: {str(response2.get('content', ''))[:100]}...")

                # Print full responses for debugging
                print(f"\n{self.agent1.agent_name} [{response1.get('message_type', 'unknown')}]")
                print(str(response1.get('content', '')))
                print(f"Confidence {float(response1.get('confidence', 0.0)):.2f}")

                print(f"\n{self.agent2.agent_name} [{response2.get('message_type', 'unknown')}]")
                print(str(response2.get('content', '')))
                print(f"Confidence {float(response2.get('confidence', 0.0)):.2f}")
                print("-" * 60)

                # Check for termination conditions
                if self._should_terminate_negotiation(response1, response2):
                    logger.info("Negotiation terminated based on agent responses")
                    break

                # Update phase if needed
                if (round_num + 1) % phase_transition_threshold == 0:
                    self._advance_negotiation_phase()

                # Prepare for next round
                current_message = str(response2.get("content", ""))
                current_agent = (self.agent2.agent_name
                                 if current_agent == self.agent1.agent_name
                                 else self.agent1.agent_name)

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in round {round_num + 1}: {e}")
                break

        # Finalize negotiation
        self.negotiation_state.status = "completed"
        results = self._analyze_negotiation_results()

        return results

    def _generate_context_queries(self) -> Dict[str, List[str]]:
        """Generate context queries based on current negotiation state"""
        base_queries = [self.topic]

        # Add phase-specific queries
        if self.negotiation_state.current_phase == NegotiationPhase.OPENING:
            base_queries.extend(["negotiation strategy", "diplomatic relations"])
        elif self.negotiation_state.current_phase == NegotiationPhase.INFORMATION_EXCHANGE:
            base_queries.extend(["economic impact", "security concerns"])
        elif self.negotiation_state.current_phase == NegotiationPhase.BARGAINING:
            base_queries.extend(["compromise solutions", "mutual benefits"])

        return {
            self.agent1.agent_name: base_queries,
            self.agent2.agent_name: base_queries
        }

    def _should_terminate_negotiation(
        self,
        response1: Dict[str, Any],
        response2: Dict[str, Any]
    ) -> bool:
        """Determine if negotiation should terminate"""
        # Check for acceptance
        if (response1.get("message_type") == MessageType.ACCEPTANCE.value and
                response2.get("message_type") == MessageType.ACCEPTANCE.value):
            return True

        # Check for complete rejection
        if (response1.get("message_type") == MessageType.REJECTION.value and
                response2.get("message_type") == MessageType.REJECTION.value):
            return True

        # Check for low confidence from both agents
        if (float(response1.get("confidence", 1.0)) < 0.3 and
                float(response2.get("confidence", 1.0)) < 0.3):
            return True

        return False

    def _advance_negotiation_phase(self):
        """Advance to the next negotiation phase"""
        phases = list(NegotiationPhase)
        try:
            current_index = phases.index(self.negotiation_state.current_phase)
        except ValueError:
            current_index = 0
        if current_index < len(phases) - 1:
            self.negotiation_state.current_phase = phases[current_index + 1]
            logger.info(f"Advanced to phase {self.negotiation_state.current_phase.value}")

    def _analyze_negotiation_results(self) -> Dict[str, Any]:
        """Analyze the negotiation results"""
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

        # Calculate negotiation metrics
        total_messages = len(self.negotiation_state.messages)
        if total_messages > 0:
            avg_confidence = sum(msg.confidence for msg in self.negotiation_state.messages) / total_messages
        else:
            avg_confidence = 0.0

        agent1_messages = [msg for msg in self.negotiation_state.messages if msg.from_agent == self.agent1.agent_name]
        agent2_messages = [msg for msg in self.negotiation_state.messages if msg.from_agent == self.agent2.agent_name]

        total_rounds = total_messages // 2 if total_messages >= 2 else total_messages

        results = {
            "negotiation_id": self.negotiation_state.negotiation_id,
            "topic": self.topic,
            "participants": self.negotiation_state.participants,
            "duration": self._calculate_duration(),
            "total_rounds": total_rounds,
            "final_phase": self.negotiation_state.current_phase.value,
            "status": self.negotiation_state.status,
            "agreements": agreements,
            "disagreements": disagreements,
            "proposals": proposals,
            "metrics": {
                "total_messages": total_messages,
                "average_confidence": avg_confidence,
                "agent1_message_count": len(agent1_messages),
                "agent2_message_count": len(agent2_messages),
                "agreement_ratio": (len(agreements) / max(1, (len(agreements) + len(disagreements))))
            },
            "message_history": [msg.to_dict() for msg in self.negotiation_state.messages],
            "final_positions": self.negotiation_state.positions
        }

        return results

    def _calculate_duration(self) -> str:
        """Calculate negotiation duration"""
        try:
            start_time = datetime.fromisoformat(self.negotiation_state.started_at)
            end_time = datetime.fromisoformat(self.negotiation_state.last_activity)
            duration = end_time - start_time
            return str(duration)
        except Exception:
            return "unknown"

    def save_negotiation_log(self, filename: str):
        """Save complete negotiation log to file"""
        results = self._analyze_negotiation_results()

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Negotiation log saved to {filename}")


async def setup_system():
    """Setup the negotiation system with availability checks"""
    print("🚀 Starting AI Agent Negotiation System (Free Models)")
    print("=" * 60)

    # Check Ollama availability
    ollama_client = OllamaClient()
    if not ollama_client.is_available():
        print("❌ Ollama server not running!")
        print("Please start Ollama:")
        print("1. Install from https://ollama.com/download")
        print("2. Start server: ollama serve")
        return None

    print("✅ Ollama server is running")

    # Check available models
    available_models = ollama_client.list_models()
    print(f"📋 Available models: {available_models}")

    # Recommended models in order of preference
    recommended_models = ["qwen3:8b", "llama3:8b", "mistral7b", "llama3.23b"]

    # Find best available models (try to assign two different models)
    usa_model = None
    russia_model = None

    for model in recommended_models:
        if any(model in available for available in available_models):
            if not usa_model:
                usa_model = model
            elif not russia_model and model != usa_model:
                russia_model = model
            if usa_model and russia_model:
                break

    # If we still don't have two models, try to pull recommended models
    if not usa_model or not russia_model:
        for model in recommended_models:
            if not any(model in available for available in available_models):
                print(f"🔄 Pulling recommended model {model}")
                ollama_client.pull_model(model)
                # refresh available models list
                available_models = ollama_client.list_models()
            if not usa_model and any(model in available for available in available_models):
                usa_model = model
            elif not russia_model and any(model in available for available in available_models) and model != usa_model:
                russia_model = model
            if usa_model and russia_model:
                break

    # Fallback to same model if only one available
    if usa_model and not russia_model:
        russia_model = usa_model
    elif not usa_model:
        print("❌ No suitable models found. Please run:")
        print("ollama pull qwen38b")
        print("ollama pull llama3.38b")
        return None

    print(f"🤖 Using models USA={usa_model}, Russia={russia_model}")

    # Configuration for Qdrant instances
    agent_configs = {
        "Agent_USA": {
            "host": "localhost",
            "port": 6333,
            "collection": "usa_collection",
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "Agent_Russia": {
            "host": "localhost",
            "port": 6334,  # Different port for second Qdrant instance
            "collection": "russia_collection",
            "embedding_model": "all-MiniLM-L6-v2"
        }
    }

    try:
        # Initialize RAG system
        print("🔄 Initializing RAG system...")
        rag_system = MultiAgentRAGSystem(agent_configs)

        # Create negotiation agents
        print("🤖 Creating negotiation agents...")

        agent_usa = RAGEnhancedNegotiationAgent(
            agent_name="Agent_USA",
            agent_description=(
                "You represent the United States perspective in international negotiations.\n\n"
                "KEY CHARACTERISTICS\n"
                "- Prioritize democratic values and international law\n"
                "- Focus on NATO alliance strength and collective security\n"
                "- Emphasize economic sanctions as policy tools\n"
                "- Support Ukraine's territorial integrity and sovereignty\n"
                "- Promote free market principles and trade relationships\n\n"
                "NEGOTIATION STYLE\n"
                "- Direct and fact-based communication\n"
                "- Willing to compromise on secondary issues\n"
                "- Firm on core democratic principles\n"
                "- Seeks multilateral solutions through international institutions"
            ),
            rag_system=rag_system,
            model=usa_model,
            ollama_client=ollama_client
        )

        agent_russia = RAGEnhancedNegotiationAgent(
            agent_name="Agent_Russia",
            agent_description=(
                "You represent the Russian perspective in international negotiations.\n\n"
                "KEY CHARACTERISTICS\n"
                "- Prioritize national security and sphere of influence\n"
                "- Emphasize historical ties and regional stability\n"
                "- Focus on economic partnerships and energy security\n"
                "- Advocate for multipolar world order\n"
                "- Highlight Western expansion concerns\n\n"
                "NEGOTIATION STYLE\n"
                "- Strategic and historically-informed communication\n"
                "- Seek recognition of legitimate security interests\n"
                "- Propose bilateral solutions and direct dialogue\n"
                "- Emphasize mutual economic benefits\n"
                "- Reference historical precedents and agreements"
            ),
            rag_system=rag_system,
            model=russia_model,
            ollama_client=ollama_client
        )

        return agent_usa, agent_russia, rag_system

    except Exception as e:
        print(f"❌ Error setting up system: {e}")
        return None


async def main():
    """Example usage of the negotiation system with free models"""
    # Setup system
    setup_result = await setup_system()
    if not setup_result:
        return

    agent_usa, agent_russia, rag_system = setup_result

    # Create negotiation manager
    print("📋 Setting up negotiation on Ukraine conflict resolution...")

    negotiation_manager = NegotiationManager(
        agent1=agent_usa,
        agent2=agent_russia,
        topic="Ukraine conflict resolution and future security arrangements"
    )

    # Define initial negotiation prompt
    initial_prompt = (
        "We are here to discuss a potential framework for resolving the ongoing conflict in Ukraine "
        "and establishing future security arrangements in Eastern Europe.\n\n"
        "Both parties should present their key concerns, priorities, and potential areas for compromise.\n\n"
        "The goal is to find a mutually acceptable path forward that addresses legitimate security concerns "
        "while respecting international law and sovereignty principles.\n\n"
        "Please begin by stating your primary position and key requirements for any potential agreement."
    )

    print("🎯 Starting negotiation...")
    print(f"Topic: {negotiation_manager.topic}")
    print(f"Participants: {', '.join(negotiation_manager.negotiation_state.participants)}")
    print(f"Models: {agent_usa.model} vs {agent_russia.model}")

    # Run the negotiation
    results = await negotiation_manager.run_full_negotiation(
        initial_prompt=initial_prompt,
        max_rounds=6,
        phase_transition_threshold=2
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"negotiation_log_{timestamp}.json"
    negotiation_manager.save_negotiation_log(log_filename)

    # Print summary
    print("\n📊 NEGOTIATION SUMMARY")
    print("=" * 40)
    print(f"Duration: {results.get('duration')}")
    print(f"Total rounds: {results.get('total_rounds')}")
    print(f"Final phase: {results.get('final_phase')}")
    print(f"Status: {results.get('status')}")
    print(f"Agreements reached: {len(results.get('agreements', []))}")
    print(f"Disagreements: {len(results.get('disagreements', []))}")
    print(f"Average confidence: {results.get('metrics', {}).get('average_confidence', 0.0):.2f}")
    print(f"Agreement ratio: {results.get('metrics', {}).get('agreement_ratio', 0.0):.2f}")

    if results.get('agreements'):
        print("\n✅ KEY AGREEMENTS")
        for i, agreement in enumerate(results['agreements'], 1):
            print(f"{i}. {agreement[:150]}...")

    if results.get('disagreements'):
        print("\n❌ MAIN DISAGREEMENTS")
        for i, disagreement in enumerate(results['disagreements'], 1):
            print(f"{i}. {disagreement[:150]}...")

    print(f"\n📄 Full log saved to {log_filename}")

    # Model performance summary
    print("\n🤖 MODEL PERFORMANCE")
    print(f"Agent USA ({agent_usa.model}): {results['metrics'].get('agent1_message_count', 0)} messages")
    print(f"Agent Russia ({agent_russia.model}): {results['metrics'].get('agent2_message_count', 0)} messages")


def check_prerequisites() -> bool:
    """Check if all prerequisites are met"""
    print("🔍 Checking Prerequisites...")

    issues: List[str] = []

    # Check Ollama
    ollama_client = OllamaClient()
    if not ollama_client.is_available():
        issues.append("❌ Ollama server not running")
        issues.append("   Fix: Install from https://ollama.com/download and run 'ollama serve'")
    else:
        print("✅ Ollama server is running")

        # Check models
        models = ollama_client.list_models()
        recommended = ["qwen3:8b", "llama3:8b", "mistral7b"]
        available_recommended = [model for model in recommended if any(model in m for m in models)]

        if not available_recommended:
            issues.append("❌ No recommended models found")
            issues.append("   Fix: Run 'ollama pull qwen3:8b' or 'ollama pull llama3:8b'")
        else:
            print(f"✅ Found models: {available_recommended}")

    # Check Python packages
    required_packages = ["requests", "asyncio"]
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            issues.append(f"❌ {package} not installed")
            issues.append(f"   Fix: pip install {package}")

    if issues:
        print("\n⚠️ ISSUES FOUND")
        for issue in issues:
            print(issue)
        print("\nPlease resolve these issues before running the negotiation system.")
        return False
    else:
        print("\n✅ All prerequisites met! Ready to run negotiations.")
        return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Just check prerequisites
        check_prerequisites()
    else:
        # Run the negotiation
        if check_prerequisites():
            try:
                asyncio.run(main())
            except KeyboardInterrupt:
                print("\nExecution interrupted by user.")
        else:
            print("\n💡 Tip: Run 'python negotiation_agents.py --check' to check prerequisites only")
