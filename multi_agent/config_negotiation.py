#!/usr/bin/env python3
"""
Enhanced Configuration System for AI Negotiation
Advanced configuration with personality analysis, batch testing, and result comparison

Usage:
python config_negotiation.py --help
python config_negotiation.py run --scenario ukraine --usa dove --russia hawk
python config_negotiation.py batch --preset-comparison
python config_negotiation.py analyze results/
"""

import json
import argparse
import asyncio
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import datetime
import os


@dataclass
class EnhancedNegotiationConfig:
    """Enhanced negotiation configuration with advanced options"""

    scenario_name: str = "ukraine"
    topic: str = "Ukraine conflict resolution"
    initial_prompt: str = ""
    context_queries: List[str] = field(default_factory=list)

    usa_personality: str = "dove"
    russia_personality: str = "hawk"
    usa_model: str = "qwen3:8b"
    russia_model: str = "llama3:8b"

    max_rounds: int = 8
    phase_transition_threshold: int = 2
    temperature: float = 0.7
    max_tokens: int = 1000
    include_opposing_views: bool = True

    rag_top_k: int = 3
    rag_score_threshold: float = 0.6
    max_context_length: int = 2000

    enable_personality_adaptation: bool = False
    confidence_threshold: float = 0.5
    adaptive_phase_transitions: bool = True
    cross_perspective_analysis: bool = True

    save_detailed_logs: bool = True
    enable_live_monitoring: bool = False
    output_directory: str = "results"
    experiment_name: str = ""

    run_multiple_iterations: int = 1
    compare_with_baselines: bool = False
    generate_personality_matrix: bool = False

    def __post_init__(self):
        if not self.context_queries:
            self.context_queries = []
        if not self.experiment_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.scenario_name}_{self.usa_personality}v{self.russia_personality}_{timestamp}"


# Enhanced personality definitions with detailed traits
ENHANCED_PERSONALITIES = {
    "hawk": {
        "description": """
        Security-focused hardliner who prioritizes national interests above compromise.

        CORE TRAITS:
        - Take strong initial positions and defend them vigorously
        - Demand significant concessions before offering any
        - Use historical precedents to justify tough stances
        - View negotiations as zero-sum competitions
        - Frame issues in terms of national security and strength

        COMMUNICATION STYLE:
        - Direct and assertive language with confidence
        - Reference military capabilities and deterrence
        - Emphasize historical grievances and threats
        - Use conditional language for any concessions
        - Focus on worst-case scenarios and risks

        STRATEGIC APPROACH:
        - Start with maximum demands to establish strong position
        - Make minimal concessions only when absolutely necessary
        - Use time pressure and deadlines as negotiation tools
        - Emphasize consequences of failing to meet demands
        """,
        "traits": ["assertive", "security_focused", "history_oriented", "minimal_compromise", "deterrence_based"],
        "effectiveness": {
            "vs_hawk": "High conflict, slow progress, potential deadlock",
            "vs_dove": "Asymmetric advantage, likely gains concessions",
            "vs_economist": "Security vs economics tension, moderate success",
            "vs_legalist": "Disputes over legal interpretations, structured conflict",
            "vs_innovator": "Traditional vs creative approaches, unpredictable"
        }
    },

    "dove": {
        "description": """
        Diplomatic mediator focused on finding mutually beneficial solutions.

        CORE TRAITS:
        - Seek collaborative solutions and shared interests
        - Offer incremental concessions to build momentum
        - Emphasize long-term relationship benefits
        - Frame issues as collaborative problem-solving
        - Show flexibility on implementation details

        COMMUNICATION STYLE:
        - Collaborative and solution-oriented language
        - Reference mutual benefits and shared goals
        - Use inclusive pronouns (we, us, our)
        - Acknowledge legitimate concerns of all parties
        - Propose phased or gradual implementation

        STRATEGIC APPROACH:
        - Build trust through early small concessions
        - Focus on win-win solutions and mutual gains
        - Use active listening and empathy building
        - Seek creative compromises and package deals
        """,
        "traits": ["collaborative", "flexible", "relationship_focused", "compromise_willing", "empathetic"],
        "effectiveness": {
            "vs_hawk": "May concede too much, but can defuse tensions",
            "vs_dove": "Quick agreements, potentially suboptimal outcomes",
            "vs_economist": "Cooperative on economic benefits, good synergy",
            "vs_legalist": "Respectful of legal frameworks, smooth negotiations",
            "vs_innovator": "Open to creative solutions, excellent synergy"
        }
    },

    "economist": {
        "description": """
        Pragmatic negotiator who prioritizes economic benefits and quantifiable outcomes.

        CORE TRAITS:
        - Focus on economic impacts and trade opportunities
        - Use data and statistical analysis in arguments
        - Willing to compromise on political issues for economic gains
        - Prefer phased implementation with economic incentives
        - Show flexibility when presented with profitable alternatives

        COMMUNICATION STYLE:
        - Data-driven and analytical language
        - Reference trade statistics and economic projections
        - Emphasize market opportunities and competitive advantages
        - Propose economic incentives and business partnerships
        - Frame agreements in terms of ROI and growth potential

        STRATEGIC APPROACH:
        - Lead with economic benefits and cost-benefit analysis
        - Use market data to support negotiation positions
        - Propose economic incentives to sweeten deals
        - Focus on measurable outcomes and performance metrics
        """,
        "traits": ["data_driven", "pragmatic", "trade_focused", "incentive_oriented", "ROI_focused"],
        "effectiveness": {
            "vs_hawk": "Economic incentives may soften security concerns",
            "vs_dove": "Economic focus complements cooperative approach",
            "vs_economist": "Data-driven negotiations, efficient outcomes",
            "vs_legalist": "Economic benefits within legal frameworks",
            "vs_innovator": "Economic analysis of innovative solutions"
        }
    },

    "legalist": {
        "description": """
        Process-oriented negotiator who emphasizes legal frameworks and institutional legitimacy.

        CORE TRAITS:
        - Reference international law and treaty precedents
        - Insist on legally binding agreements with enforcement
        - Use formal diplomatic protocols and procedures
        - Quote specific legal frameworks and court decisions
        - Show flexibility only within established legal boundaries

        COMMUNICATION STYLE:
        - Formal and precise legal language
        - Reference specific treaties and legal precedents
        - Emphasize binding commitments and enforcement mechanisms
        - Propose detailed legal frameworks and procedures
        - Focus on institutional legitimacy and due process

        STRATEGIC APPROACH:
        - Build agreements on solid legal foundations
        - Use legal precedents to support positions
        - Insist on proper procedures and documentation
        - Focus on enforceability and compliance mechanisms
        """,
        "traits": ["law_focused", "formal", "precedent_oriented", "binding_agreements", "institutional"],
        "effectiveness": {
            "vs_hawk": "Legal constraints may limit aggressive tactics",
            "vs_dove": "Legal structure supports cooperative agreements",
            "vs_economist": "Legal frameworks for economic arrangements",
            "vs_legalist": "Highly structured, precedent-based negotiations",
            "vs_innovator": "Legal feasibility of innovative approaches"
        }
    },

    "innovator": {
        "description": """
        Creative problem-solver who focuses on novel approaches and experimental solutions.

        CORE TRAITS:
        - Propose unconventional solutions and creative frameworks
        - Willing to try experimental approaches and pilot programs
        - Focus on technology and innovation as solution enablers
        - Embrace calculated risks for transformative outcomes
        - Challenge traditional approaches with new paradigms

        COMMUNICATION STYLE:
        - Creative and forward-thinking language
        - Reference emerging technologies and future trends
        - Propose experimental and pilot approaches
        - Use metaphors and analogies to explain concepts
        - Emphasize adaptation and evolution over tradition

        STRATEGIC APPROACH:
        - Reframe problems to find novel solutions
        - Propose pilot programs and experimental approaches
        - Use technology to enable new forms of cooperation
        - Focus on long-term transformation over short-term gains
        """,
        "traits": ["creative", "experimental", "tech_oriented", "risk_taking", "paradigm_shifting"],
        "effectiveness": {
            "vs_hawk": "Creative solutions may address security concerns",
            "vs_dove": "Innovative approaches enhance cooperation",
            "vs_economist": "Technology-driven economic opportunities",
            "vs_legalist": "Innovation within legal frameworks",
            "vs_innovator": "Highly creative but potentially impractical solutions"
        }
    }
}

ENHANCED_SCENARIOS = {
    "ukraine": {
        "topic": "Ukraine conflict resolution and future security arrangements",
        "description": "Complex geopolitical negotiation involving territorial integrity, security guarantees, and regional stability",
        "initial_prompt": """
        We are conducting high-level negotiations to establish a comprehensive framework for resolving the ongoing Ukraine conflict and creating sustainable security arrangements in Eastern Europe.

        NEGOTIATION MANDATE:
        Both parties have been authorized to explore all reasonable options for achieving lasting peace while protecting their core national interests.

        KEY NEGOTIATION AREAS:
        1. Territorial arrangements and sovereignty principles
        2. Security guarantees and alliance relationships
        3. Economic reconstruction and cooperation frameworks
        4. Humanitarian support and refugee integration
        5. Future diplomatic engagement and conflict prevention mechanisms

        CONSTRAINTS AND REQUIREMENTS:
        - Solutions must be realistic and implementable
        - Both parties must achieve meaningful gains
        - International law and democratic principles should be respected
        - Economic costs and benefits must be carefully considered
        - Long-term stability is the ultimate objective

        Please present your opening position, including your primary objectives, key requirements, and potential areas where your side might show flexibility. Be prepared to engage in substantive discussions based on factual information and strategic analysis.
        """,
        "context_queries": [
            "Ukraine conflict history", "security guarantees", "territorial integrity",
            "NATO expansion", "economic sanctions", "diplomatic negotiations",
            "Eastern European security", "international law", "peace agreements"
        ],
        "complexity_level": "high",
        "typical_duration": "8-12 rounds"
    },

    "climate": {
        "topic": "International climate cooperation and emissions reduction framework",
        "description": "Global environmental negotiation balancing economic development with climate action",
        "initial_prompt": """
        We are negotiating a comprehensive bilateral climate cooperation agreement that addresses the urgent challenge of global warming while respecting each nation's development needs and economic realities.

        NEGOTIATION MANDATE:
        Develop actionable commitments that significantly contribute to global climate goals while ensuring fair burden-sharing and economic competitiveness.

        KEY NEGOTIATION AREAS:
        1. National emissions reduction targets and timelines
        2. Technology transfer and green innovation partnerships
        3. Climate finance commitments and funding mechanisms
        4. Monitoring, reporting, and verification systems
        5. Support for developing nations and just transition policies

        CONSTRAINTS AND REQUIREMENTS:
        - Commitments must be ambitious yet achievable
        - Economic impacts on key industries must be considered
        - Technology sharing must respect intellectual property
        - Financial commitments must be sustainable
        - Agreement must enhance rather than undermine competitiveness

        Present your climate policy priorities, acceptable commitment levels, and preferred mechanisms for international cooperation. Consider both environmental urgency and economic practicalities.
        """,
        "context_queries": [
            "climate change impacts", "emissions targets", "green technology",
            "climate finance", "carbon pricing", "renewable energy",
            "just transition", "climate adaptation", "international climate agreements"
        ],
        "complexity_level": "high",
        "typical_duration": "6-10 rounds"
    },

    "trade": {
        "topic": "Bilateral trade agreement and economic partnership",
        "description": "Economic negotiation focusing on market access, tariffs, and trade facilitation",
        "initial_prompt": """
        We are negotiating a comprehensive bilateral trade agreement designed to boost economic growth, create jobs, and enhance competitiveness for both nations.

        NEGOTIATION MANDATE:
        Create a mutually beneficial trade relationship that expands market access while protecting sensitive domestic industries and maintaining regulatory sovereignty.

        KEY NEGOTIATION AREAS:
        1. Tariff reductions and elimination schedules for goods
        2. Services market access and professional mobility
        3. Digital trade rules and data governance
        4. Intellectual property protections and enforcement
        5. Investment protections and dispute resolution mechanisms
        6. Labor and environmental standards coordination

        CONSTRAINTS AND REQUIREMENTS:
        - Protect key domestic industries and employment
        - Ensure fair competition and reciprocal access
        - Maintain regulatory autonomy in sensitive areas
        - Include enforceable labor and environmental standards
        - Provide effective dispute resolution mechanisms

        Outline your trade priorities, sensitive sectors requiring protection, and preferred approaches to market opening. Focus on creating value while managing domestic political constraints.
        """,
        "context_queries": [
            "trade statistics", "tariff impacts", "export opportunities",
            "industry competitiveness", "trade agreements", "market access",
            "intellectual property", "digital trade", "investment protection"
        ],
        "complexity_level": "medium",
        "typical_duration": "6-8 rounds"
    },

    "cybersecurity": {
        "topic": "Cybersecurity cooperation and digital governance framework",
        "description": "Technology-focused negotiation addressing digital threats and governance challenges",
        "initial_prompt": """
        We are developing a bilateral cybersecurity cooperation agreement to address growing digital threats while establishing norms for responsible state behavior in cyberspace.

        NEGOTIATION MANDATE:
        Create effective cooperation mechanisms that enhance cybersecurity while respecting sovereignty, privacy rights, and legitimate national security interests.

        KEY NEGOTIATION AREAS:
        1. Cybercrime prevention and law enforcement cooperation
        2. Critical infrastructure protection and information sharing
        3. Threat intelligence exchange protocols and standards
        4. Digital sovereignty principles and data governance
        5. Norms for responsible state behavior in cyberspace
        6. Capacity building and technical assistance programs

        CONSTRAINTS AND REQUIREMENTS:
        - Balance security cooperation with sovereignty concerns
        - Protect sensitive national security information
        - Respect privacy rights and civil liberties
        - Ensure reciprocal benefits and burden-sharing
        - Create enforceable cooperation mechanisms

        Present your cybersecurity priorities, cooperation boundaries, and preferred mechanisms for information sharing and joint action. Address both technical capabilities and policy frameworks.
        """,
        "context_queries": [
            "cybersecurity threats", "digital governance", "cyber norms",
            "data protection", "critical infrastructure", "cyber crime",
            "information sharing", "digital sovereignty", "cyber warfare"
        ],
        "complexity_level": "medium",
        "typical_duration": "5-7 rounds"
    },

    "energy": {
        "topic": "Energy security and transition cooperation agreement",
        "description": "Strategic negotiation balancing energy security with transition to sustainable energy",
        "initial_prompt": """
        We are negotiating an energy cooperation framework that addresses immediate security needs while facilitating the transition to sustainable energy systems.

        NEGOTIATION MANDATE:
        Develop energy partnerships that enhance security, affordability, and sustainability while managing the complex transition from fossil fuels to renewable energy.

        KEY NEGOTIATION AREAS:
        1. Energy supply diversification and security guarantees
        2. Renewable energy technology cooperation and investment
        3. Traditional energy resource management and pricing
        4. Grid integration and infrastructure development
        5. Just transition support for energy-dependent communities
        6. Energy efficiency and demand management programs

        CONSTRAINTS AND REQUIREMENTS:
        - Ensure reliable and affordable energy supplies
        - Support clean energy transition goals
        - Protect energy-dependent workers and communities
        - Maintain energy infrastructure investments
        - Balance environmental and economic objectives

        Outline your energy security priorities, transition timelines, and preferred cooperation mechanisms. Consider both immediate needs and long-term sustainability goals.
        """,
        "context_queries": [
            "energy security", "renewable energy", "energy transition",
            "pipeline infrastructure", "energy markets", "grid integration",
            "just transition", "energy efficiency", "fossil fuels"
        ],
        "complexity_level": "medium",
        "typical_duration": "6-8 rounds"
    }
}

# Preset configurations for systematic testing
ENHANCED_PRESETS = {
    "diplomatic_cooperation": {
        "description": "Both sides focused on diplomatic solutions and cooperation",
        "scenario": "ukraine",
        "usa_personality": "dove",
        "russia_personality": "dove",
        "expected_outcome": "Quick agreements, potentially suboptimal for both parties",
        "research_value": "Baseline for cooperative negotiations"
    },

    "hardline_confrontation": {
        "description": "Security-focused hardliners with minimal compromise",
        "scenario": "ukraine",
        "usa_personality": "hawk",
        "russia_personality": "hawk",
        "expected_outcome": "Prolonged negotiations, potential deadlock, few concessions",
        "research_value": "Stress test for conflict resolution mechanisms"
    },

    "asymmetric_power": {
        "description": "Aggressive negotiator vs cooperative partner",
        "scenario": "ukraine",
        "usa_personality": "hawk",
        "russia_personality": "dove",
        "expected_outcome": "Asymmetric outcomes favoring hawk, quick resolution",
        "research_value": "Study of power dynamics and concession patterns"
    },

    "economic_pragmatism": {
        "description": "Economic interests drive negotiation strategies",
        "scenario": "trade",
        "usa_personality": "economist",
        "russia_personality": "economist",
        "expected_outcome": "Data-driven agreements, efficient negotiations",
        "research_value": "Rational actor model testing"
    },

    "institutional_framework": {
        "description": "Legal frameworks and institutional legitimacy focus",
        "scenario": "climate",
        "usa_personality": "legalist",
        "russia_personality": "legalist",
        "expected_outcome": "Highly structured agreements, emphasis on enforcement",
        "research_value": "Role of legal frameworks in international negotiations"
    },

    "innovation_focused": {
        "description": "Creative problem-solving and experimental approaches",
        "scenario": "cybersecurity",
        "usa_personality": "innovator",
        "russia_personality": "innovator",
        "expected_outcome": "Novel solutions, experimental approaches, high creativity",
        "research_value": "Innovation in diplomatic problem-solving"
    },

    "mixed_realism": {
        "description": "Realistic mix of security concerns and legal constraints",
        "scenario": "ukraine",
        "usa_personality": "legalist",
        "russia_personality": "hawk",
        "expected_outcome": "Structured conflict, emphasis on precedents vs security",
        "research_value": "Law vs security trade-offs in negotiations"
    },

    "pragmatic_innovation": {
        "description": "Economic pragmatism meets creative problem-solving",
        "scenario": "energy",
        "usa_personality": "economist",
        "russia_personality": "innovator",
        "expected_outcome": "Cost-effective innovative solutions, technology focus",
        "research_value": "Integration of economic and innovation perspectives"
    }
}


class EnhancedConfigurationManager:
    """Advanced configuration manager with analysis and batch testing capabilities"""

    def __init__(self):
        self.personalities = ENHANCED_PERSONALITIES
        self.scenarios = ENHANCED_SCENARIOS
        self.presets = ENHANCED_PRESETS

    def create_config(self, **kwargs) -> EnhancedNegotiationConfig:
        """Create enhanced negotiation configuration with validation"""

        usa_personality = kwargs.get('usa_personality', 'dove')
        russia_personality = kwargs.get('russia_personality', 'hawk')
        scenario = kwargs.get('scenario', 'ukraine')

        if usa_personality not in self.personalities:
            raise ValueError(
                f"Unknown USA personality: {usa_personality}. Available: {list(self.personalities.keys())}")
        if russia_personality not in self.personalities:
            raise ValueError(
                f"Unknown Russia personality: {russia_personality}. Available: {list(self.personalities.keys())}")
        if scenario not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(self.scenarios.keys())}")

        scenario_data = self.scenarios[scenario]

        config = EnhancedNegotiationConfig(
            scenario_name=scenario,
            topic=scenario_data["topic"],
            initial_prompt=scenario_data["initial_prompt"],
            context_queries=scenario_data["context_queries"],
            usa_personality=usa_personality,
            russia_personality=russia_personality,
            **kwargs
        )

        return config

    def create_preset_config(self, preset_name: str, **overrides) -> EnhancedNegotiationConfig:
        """Create configuration from preset with optional overrides"""
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(self.presets.keys())}")

        preset = self.presets[preset_name]

        # Merge preset with overrides
        config_params = {
            "scenario": preset["scenario"],
            "usa_personality": preset["usa_personality"],
            "russia_personality": preset["russia_personality"],
            **overrides
        }

        return self.create_config(**config_params)

    # def generate_personality_matrix(self, scenario: str = "ukraine") -> List[EnhancedNegotiationConfig]:
    #     """Generate all personality combinations for systematic testing"""
    #     personalities = list(self.personalities.keys())
    #     configs = []
    #
    #     for usa_p in personalities:
    #         for russia_p in personalities:
    #             config = self.create_config(
    #                 scenario=scenario,
    #                 usa_personality=usa_p,
    #                 russia_personality=russia_p,
    #                 experiment_name=f"matrix_{scenario}_{usa_p}v{russia_p}"
    #             )
    #             configs.append(config)
    #
    #     return configs

    # def analyze_personality_compatibility(self, usa_personality: str, russia_personality: str) -> Dict[str, Any]:
    #     """Analyze compatibility between personality types"""
    #     usa_traits = self.personalities[usa_personality]
    #     russia_traits = self.personalities[russia_personality]
    #
    #     expected_outcome = usa_traits["effectiveness"].get(f"vs_{russia_personality}", "Unknown interaction")
    #
    #     compatibility_score = self._calculate_compatibility_score(usa_personality, russia_personality)
    #
    #     return {
    #         "usa_personality": usa_personality,
    #         "russia_personality": russia_personality,
    #         "compatibility_score": compatibility_score,
    #         "expected_outcome": expected_outcome,
    #         "usa_traits": usa_traits["traits"],
    #         "russia_traits": russia_traits["traits"],
    #         "negotiation_prediction": self._predict_negotiation_dynamics(usa_personality, russia_personality)
    #     }

    # def _calculate_compatibility_score(self, usa_p: str, russia_p: str) -> float:
    #     """Calculate numerical compatibility score (0-1)"""
    #     compatibility_matrix = {
    #         ("dove", "dove"): 0.9,
    #         ("dove", "hawk"): 0.3,
    #         ("dove", "economist"): 0.7,
    #         ("dove", "legalist"): 0.8,
    #         ("dove", "innovator"): 0.9,
    #         ("hawk", "hawk"): 0.2,
    #         ("hawk", "economist"): 0.5,
    #         ("hawk", "legalist"): 0.4,
    #         ("hawk", "innovator"): 0.6,
    #         ("economist", "economist"): 0.8,
    #         ("economist", "legalist"): 0.7,
    #         ("economist", "innovator"): 0.8,
    #         ("legalist", "legalist"): 0.7,
    #         ("legalist", "innovator"): 0.6,
    #         ("innovator", "innovator"): 0.8
    #     }
    #
    #     score = compatibility_matrix.get((usa_p, russia_p)) or compatibility_matrix.get((russia_p, usa_p))
    #     return score if score is not None else 0.5

    # def _predict_negotiation_dynamics(self, usa_p: str, russia_p: str) -> Dict[str, str]:
    #     """Predict likely negotiation characteristics"""
    #     predictions = {
    #         ("dove", "dove"): {
    #             "pace": "Fast",
    #             "conflict_level": "Low",
    #             "agreement_likelihood": "High",
    #             "outcome_quality": "Suboptimal but stable"
    #         },
    #         ("hawk", "hawk"): {
    #             "pace": "Slow",
    #             "conflict_level": "High",
    #             "agreement_likelihood": "Low",
    #             "outcome_quality": "Either breakthrough or deadlock"
    #         },
    #         ("dove", "hawk"): {
    #             "pace": "Medium",
    #             "conflict_level": "Medium",
    #             "agreement_likelihood": "Medium",
    #             "outcome_quality": "Asymmetric favoring hawk"
    #         },
    #         ("economist", "economist"): {
    #             "pace": "Fast",
    #             "conflict_level": "Low",
    #             "agreement_likelihood": "High",
    #             "outcome_quality": "Efficient and data-driven"
    #         }
    #     }
    #
    #     key = (usa_p, russia_p)
    #     return predictions.get(key, predictions.get((russia_p, usa_p), {
    #         "pace": "Variable",
    #         "conflict_level": "Medium",
    #         "agreement_likelihood": "Medium",
    #         "outcome_quality": "Depends on scenario"
    #     }))

    # def save_config(self, config: EnhancedNegotiationConfig, filename: str):
    #     """Save configuration with metadata"""
    #     config_dict = asdict(config)
    #
    #     # Add metadata
    #     config_dict["_metadata"] = {
    #         "created_at": datetime.datetime.now().isoformat(),
    #         "personality_analysis": self.analyze_personality_compatibility(
    #             config.usa_personality, config.russia_personality
    #         ),
    #         "scenario_complexity": self.scenarios[config.scenario_name].get("complexity_level", "medium"),
    #         "expected_duration": self.scenarios[config.scenario_name].get("typical_duration", "6-8 rounds")
    #     }
    #
    #     # Ensure directory exists
    #     Path(filename).parent.mkdir(parents=True, exist_ok=True)
    #
    #     with open(filename, 'w', encoding='utf-8') as f:
    #         json.dump(config_dict, f, indent=2, ensure_ascii=False)
    #
    #     print(f"Configuration saved to {filename}")
    #
    # def load_config(self, filename: str) -> EnhancedNegotiationConfig:
    #     """Load configuration from file"""
    #     with open(filename, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    #
    #     # Remove metadata for config creation
    #     data.pop('_metadata', None)
    #
    #     return EnhancedNegotiationConfig(**data)


# class BatchTestingManager:
#     """Manager for running batch tests and comparative analysis"""
#
#     def __init__(self, config_manager: EnhancedConfigurationManager):
#         self.config_manager = config_manager
#
#     async def run_preset_comparison(self, output_dir: str = "batch_results") -> Dict[str, Any]:
#         """Run all preset configurations and compare results"""
#         print("Running preset comparison batch test...")
#
#         results = {}
#         Path(output_dir).mkdir(parents=True, exist_ok=True)
#
#         for preset_name, preset_info in self.config_manager.presets.items():
#             print(f"\nRunning preset: {preset_name}")
#             print(f"Description: {preset_info['description']}")
#
#             try:
#                 config = self.config_manager.create_preset_config(preset_name)
#                 config.output_directory = output_dir
#                 config.experiment_name = f"preset_{preset_name}"
#
#                 # Import and run negotiation
#                 from negotiation_agents import run_configured_negotiation
#                 result = await run_configured_negotiation(config)
#
#                 results[preset_name] = {
#                     "config": asdict(config),
#                     "result": result,
#                     "preset_info": preset_info
#                 }
#
#                 print(f"Completed {preset_name}: {result.get('status', 'unknown')}")
#
#             except Exception as e:
#                 print(f"Error running preset {preset_name}: {e}")
#                 results[preset_name] = {"error": str(e)}
#
#         # Save batch results
#         batch_filename = f"{output_dir}/batch_preset_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#         with open(batch_filename, 'w') as f:
#             json.dump(results, f, indent=2, default=str)
#
#         print(f"\nBatch test completed. Results saved to {batch_filename}")
#         return results
#
#     async def run_personality_matrix(self, scenario: str = "ukraine", output_dir: str = "matrix_results") -> Dict[
#         str, Any]:
#         """Run full personality matrix for systematic analysis"""
#         print(f"Running personality matrix for scenario: {scenario}")
#
#         configs = self.config_manager.generate_personality_matrix(scenario)
#         results = {}
#         Path(output_dir).mkdir(parents=True, exist_ok=True)
#
#         total_configs = len(configs)
#         for i, config in enumerate(configs, 1):
#             personality_pair = f"{config.usa_personality}_vs_{config.russia_personality}"
#             print(f"\nRunning {i}/{total_configs}: {personality_pair}")
#
#             try:
#                 config.output_directory = output_dir
#
#                 from negotiation_agents import run_configured_negotiation
#                 result = await run_configured_negotiation(config)
#
#                 results[personality_pair] = {
#                     "config": asdict(config),
#                     "result": result
#                 }
#
#                 print(
#                     f"Completed {personality_pair}: {result.get('outcomes', {}).get('agreement_ratio', 0):.2f} agreement ratio")
#
#             except Exception as e:
#                 print(f"Error running {personality_pair}: {e}")
#                 results[personality_pair] = {"error": str(e)}
#
#         # Save matrix results
#         matrix_filename = f"{output_dir}/personality_matrix_{scenario}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#         with open(matrix_filename, 'w') as f:
#             json.dump(results, f, indent=2, default=str)
#
#         print(f"\nPersonality matrix completed. Results saved to {matrix_filename}")
#         return results


def print_enhanced_summary(config: EnhancedNegotiationConfig):
    """Print detailed configuration summary"""
    print("\n" + "=" * 80)
    print("ENHANCED NEGOTIATION CONFIGURATION")
    print("=" * 80)

    print(f"\nSCENARIO: {config.scenario_name}")
    print(f"Topic: {config.topic}")
    scenario_info = ENHANCED_SCENARIOS.get(config.scenario_name, {})
    if scenario_info:
        print(f"Complexity: {scenario_info.get('complexity_level', 'medium')}")
        print(f"Expected Duration: {scenario_info.get('typical_duration', '6-8 rounds')}")

    print(f"\nAGENTS:")
    print(f"  USA: {config.usa_personality} personality, {config.usa_model} model")
    print(f"  Russia: {config.russia_personality} personality, {config.russia_model} model")

    # Personality analysis
    config_manager = EnhancedConfigurationManager()
    # compatibility = config_manager.analyze_personality_compatibility(
    #     config.usa_personality, config.russia_personality
    # )

    # print(f"\nPERSONALITY ANALYSIS:")
    # print(f"  Compatibility Score: {compatibility['compatibility_score']:.2f}/1.0")
    # print(f"  Expected Pace: {compatibility['negotiation_prediction'].get('pace', 'Unknown')}")
    # print(f"  Conflict Level: {compatibility['negotiation_prediction'].get('conflict_level', 'Unknown')}")
    # print(f"  Agreement Likelihood: {compatibility['negotiation_prediction'].get('agreement_likelihood', 'Unknown')}")

    print(f"\nNEGOTIATION PARAMETERS:")
    print(f"  Max Rounds: {config.max_rounds}")
    print(f"  Phase Transition: Every {config.phase_transition_threshold} rounds")
    print(f"  Temperature: {config.temperature}")
    print(f"  Include Opposing Views: {'Yes' if config.include_opposing_views else 'No'}")

    print(f"\nRAG SETTINGS:")
    print(f"  Top-K Retrieval: {config.rag_top_k}")
    print(f"  Score Threshold: {config.rag_score_threshold}")
    print(f"  Max Context Length: {config.max_context_length}")

    print(f"\nOUTPUT:")
    print(f"  Output Directory: {config.output_directory}")
    print(f"  Experiment Name: {config.experiment_name}")
    print(f"  Save Detailed Logs: {'Yes' if config.save_detailed_logs else 'No'}")

    print("=" * 80)


async def main():
    """Enhanced main function with comprehensive command-line interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced AI Negotiation Configuration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Single negotiation with specific personalities
  python enhanced_config_negotiation.py run --scenario ukraine --usa dove --russia hawk

  # Use a preset configuration
  python enhanced_config_negotiation.py run --preset diplomatic_cooperation

  # Run batch comparison of all presets
  python enhanced_config_negotiation.py batch --preset-comparison

  # Generate personality matrix for systematic analysis
  python enhanced_config_negotiation.py batch --personality-matrix --scenario ukraine

  # Analyze personality compatibility
  python enhanced_config_negotiation.py analyze --usa-personality dove --russia-personality hawk

  # Show available options
  python enhanced_config_negotiation.py list --personalities
  python enhanced_config_negotiation.py list --scenarios
  python enhanced_config_negotiation.py list --presets
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # RUN command - single negotiation
    run_parser = subparsers.add_parser('run', help='Run single negotiation')
    run_group = run_parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument('--preset', help='Use preset configuration')
    run_group.add_argument('--custom', action='store_true', help='Use custom configuration')

    # Configuration options for custom runs
    run_parser.add_argument('--scenario', default='ukraine', help='Negotiation scenario')
    run_parser.add_argument('--usa', dest='usa_personality', default='dove', help='USA agent personality')
    run_parser.add_argument('--russia', dest='russia_personality', default='hawk', help='Russia agent personality')
    run_parser.add_argument('--usa-model', default='qwen3:8b', help='USA agent model')
    run_parser.add_argument('--russia-model', default='llama3:8b', help='Russia agent model')
    run_parser.add_argument('--rounds', type=int, default=8, help='Maximum rounds')
    run_parser.add_argument('--temperature', type=float, default=0.7, help='Model creativity (0.1-1.0)')
    run_parser.add_argument('--opposing-views', action='store_true', help='Include opposing perspectives')
    run_parser.add_argument('--output-dir', default='results', help='Output directory')
    run_parser.add_argument('--experiment-name', help='Custom experiment name')

    # BATCH command - multiple negotiations
    batch_parser = subparsers.add_parser('batch', help='Run batch testing')
    batch_group = batch_parser.add_mutually_exclusive_group(required=True)
    batch_group.add_argument('--preset-comparison', action='store_true', help='Compare all presets')
    batch_group.add_argument('--personality-matrix', action='store_true', help='Run personality matrix')
    batch_group.add_argument('--custom-batch', help='JSON file with custom batch configurations')

    batch_parser.add_argument('--scenario', default='ukraine', help='Scenario for matrix testing')
    batch_parser.add_argument('--output-dir', default='batch_results', help='Batch output directory')
    batch_parser.add_argument('--parallel', type=int, default=1, help='Parallel execution count')

    # ANALYZE command - configuration analysis
    analyze_parser = subparsers.add_parser('analyze', help='Analyze configurations or results')
    analyze_group = analyze_parser.add_mutually_exclusive_group(required=True)
    analyze_group.add_argument('--personality-compatibility', action='store_true',
                               help='Analyze personality compatibility')
    analyze_group.add_argument('--results-dir', help='Analyze results in directory')
    analyze_group.add_argument('--config-file', help='Analyze specific configuration file')

    analyze_parser.add_argument('--usa-personality', help='USA personality for compatibility analysis')
    analyze_parser.add_argument('--russia-personality', help='Russia personality for compatibility analysis')

    # LIST command - show available options
    list_parser = subparsers.add_parser('list', help='List available options')
    list_group = list_parser.add_mutually_exclusive_group(required=True)
    list_group.add_argument('--personalities', action='store_true', help='List personality types')
    list_group.add_argument('--scenarios', action='store_true', help='List scenarios')
    list_group.add_argument('--presets', action='store_true', help='List preset configurations')
    list_group.add_argument('--all', action='store_true', help='List all options')

    # SAVE/LOAD commands
    save_parser = subparsers.add_parser('save-config', help='Save configuration to file')
    save_parser.add_argument('filename', help='Output filename')
    save_parser.add_argument('--scenario', default='ukraine')
    save_parser.add_argument('--usa', dest='usa_personality', default='dove')
    save_parser.add_argument('--russia', dest='russia_personality', default='hawk')

    load_parser = subparsers.add_parser('load-config', help='Load and run configuration from file')
    load_parser.add_argument('filename', help='Configuration filename')
    load_parser.add_argument('--override', nargs='*', help='Override specific settings (key=value)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    config_manager = EnhancedConfigurationManager()

    try:
        if args.command == 'run':
            await handle_run_command(args, config_manager)
        # elif args.command == 'batch':
        #     await handle_batch_command(args, config_manager)
        elif args.command == 'analyze':
            handle_analyze_command(args, config_manager)
        elif args.command == 'list':
            handle_list_command(args, config_manager)
        # elif args.command == 'save-config':
        #     handle_save_config_command(args, config_manager)
        # elif args.command == 'load-config':
        #     await handle_load_config_command(args, config_manager)

    except Exception as e:
        print(f"Error: {e}")
        return 1


async def handle_run_command(args, config_manager):
    """Handle single negotiation run"""
    if args.preset:
        print(f"Running preset configuration: {args.preset}")
        config = config_manager.create_preset_config(args.preset)

        # Apply any overrides
        if hasattr(args, 'output_dir') and args.output_dir != 'results':
            config.output_directory = args.output_dir
        if hasattr(args, 'experiment_name') and args.experiment_name:
            config.experiment_name = args.experiment_name

    else:
        print("Running custom configuration...")
        config = config_manager.create_config(
            scenario=args.scenario,
            usa_personality=args.usa_personality,
            russia_personality=args.russia_personality,
            usa_model=args.usa_model,
            russia_model=args.russia_model,
            max_rounds=args.rounds,
            temperature=args.temperature,
            include_opposing_views=args.opposing_views,
            output_directory=args.output_dir,
            experiment_name=args.experiment_name or None
        )

    # Print configuration summary
    print_enhanced_summary(config)

    # Confirm before running
    response = input("\nProceed with negotiation? (y/N): ").strip().lower()
    if response != 'y':
        print("Negotiation cancelled.")
        return

    print("\nStarting negotiation...")

    try:
        from negotiation_agents import run_configured_negotiation
        results = await run_configured_negotiation(config)

        print("\nNEGOTIATION COMPLETED!")
        print("=" * 50)
        print(f"Status: {results.get('status')}")
        print(f"Rounds: {results.get('total_rounds')}")
        print(f"Agreement Ratio: {results.get('outcomes', {}).get('agreement_ratio', 0):.2f}")
        print(f"Average Confidence: {results.get('metrics', {}).get('average_confidence', 0):.2f}")
        print(f"RAG Sources Used: {results.get('metrics', {}).get('unique_rag_sources', 0)}")

        if results.get('outcomes', {}).get('agreements'):
            print(f"\nKey Agreements ({len(results['outcomes']['agreements'])}):")
            for i, agreement in enumerate(results['outcomes']['agreements'][:3], 1):
                print(f"  {i}. {agreement[:100]}...")

    except ImportError:
        print("Error: improved_negotiation_agents.py not found.")
        print("Make sure the negotiation system files are in the same directory.")
    except Exception as e:
        print(f"Error running negotiation: {e}")


# async def handle_batch_command(args, config_manager):
#     """Handle batch testing commands"""
#     batch_manager = BatchTestingManager(config_manager)
#
#     if args.preset_comparison:
#         print("Running preset comparison batch test...")
#         results = await batch_manager.run_preset_comparison(args.output_dir)
#         print_batch_summary(results, "Preset Comparison")
#
#     elif args.personality_matrix:
#         print(f"Running personality matrix for scenario: {args.scenario}")
#         results = await batch_manager.run_personality_matrix(args.scenario, args.output_dir)
#         print_batch_summary(results, f"Personality Matrix - {args.scenario}")
#
#     elif args.custom_batch:
#         print(f"Running custom batch from: {args.custom_batch}")
#         # Implementation for custom batch files
#         print("Custom batch testing not yet implemented.")


def handle_analyze_command(args, config_manager):
    """Handle analysis commands"""
    if args.personality_compatibility:
        if not args.usa_personality or not args.russia_personality:
            print("Error: Both --usa-personality and --russia-personality required for compatibility analysis")
            return

        analysis = config_manager.analyze_personality_compatibility(
            args.usa_personality, args.russia_personality
        )

        print(f"\nPERSONALITY COMPATIBILITY ANALYSIS")
        print("=" * 50)
        print(f"USA: {analysis['usa_personality']} | Russia: {analysis['russia_personality']}")
        print(f"Compatibility Score: {analysis['compatibility_score']:.2f}/1.0")
        print(f"\nExpected Outcome: {analysis['expected_outcome']}")
        print(f"\nNegotiation Predictions:")
        for key, value in analysis['negotiation_prediction'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

        print(f"\nUSA Traits: {', '.join(analysis['usa_traits'])}")
        print(f"Russia Traits: {', '.join(analysis['russia_traits'])}")

    elif args.results_dir:
        print(f"Analyzing results in: {args.results_dir}")
        analyze_results_directory(args.results_dir)

    elif args.config_file:
        print(f"Analyzing configuration: {args.config_file}")
        config = config_manager.load_config(args.config_file)
        print_enhanced_summary(config)


def handle_list_command(args, config_manager):
    """Handle list commands"""
    if args.personalities or args.all:
        print("\nAVAILABLE PERSONALITIES:")
        print("=" * 50)
        for name, info in config_manager.personalities.items():
            traits = ", ".join(info['traits'])
            print(f"\n{name.upper()}:")
            print(f"  Traits: {traits}")
            print(f"  Description: {info['description'].split('.')[0]}...")

    if args.scenarios or args.all:
        print("\nAVAILABLE SCENARIOS:")
        print("=" * 50)
        for name, info in config_manager.scenarios.items():
            print(f"\n{name.upper()}:")
            print(f"  Topic: {info['topic']}")
            print(f"  Complexity: {info.get('complexity_level', 'medium')}")
            print(f"  Duration: {info.get('typical_duration', '6-8 rounds')}")

    if args.presets or args.all:
        print("\nAVAILABLE PRESETS:")
        print("=" * 50)
        for name, info in config_manager.presets.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Scenario: {info['scenario']}")
            print(f"  Personalities: {info['usa_personality']} vs {info['russia_personality']}")
            print(f"  Expected: {info['expected_outcome']}")


# def handle_save_config_command(args, config_manager):
#     """Handle save configuration command"""
#     config = config_manager.create_config(
#         scenario=args.scenario,
#         usa_personality=args.usa_personality,
#         russia_personality=args.russia_personality
#     )
#
#     config_manager.save_config(config, args.filename)
#     print_enhanced_summary(config)


# async def handle_load_config_command(args, config_manager):
#     """Handle load configuration command"""
#     config = config_manager.load_config(args.filename)
#
#     # Apply overrides if provided
#     if args.override:
#         for override in args.override:
#             if '=' in override:
#                 key, value = override.split('=', 1)
#                 if hasattr(config, key):
#                     # Convert value to appropriate type
#                     if hasattr(getattr(config, key), '__class__'):
#                         original_type = type(getattr(config, key))
#                         if original_type == bool:
#                             value = value.lower() in ('true', '1', 'yes')
#                         elif original_type in (int, float):
#                             value = original_type(value)
#                     setattr(config, key, value)
#                     print(f"Override applied: {key} = {value}")
#
#     print_enhanced_summary(config)
#
#     # Run the loaded configuration
#     response = input("\nRun loaded configuration? (y/N): ").strip().lower()
#     if response == 'y':
#         try:
#             from negotiation_agents import run_configured_negotiation
#             results = await run_configured_negotiation(config)
#             print("\nConfiguration executed successfully!")
#         except Exception as e:
#             print(f"Error running configuration: {e}")


def print_batch_summary(results: Dict[str, Any], test_type: str):
    """Print summary of batch test results"""
    print(f"\n{test_type.upper()} SUMMARY")
    print("=" * 60)

    successful = [k for k, v in results.items() if 'error' not in v]
    failed = [k for k, v in results.items() if 'error' in v]

    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\nSUCCESSFUL TESTS:")
        for test_name in successful:
            result = results[test_name].get('result', {})
            agreement_ratio = result.get('outcomes', {}).get('agreement_ratio', 0)
            avg_confidence = result.get('metrics', {}).get('average_confidence', 0)
            print(f"  {test_name}: Agreement={agreement_ratio:.2f}, Confidence={avg_confidence:.2f}")

    if failed:
        print(f"\nFAILED TESTS:")
        for test_name in failed:
            error = results[test_name].get('error', 'Unknown error')
            print(f"  {test_name}: {error}")


def analyze_results_directory(results_dir: str):
    """Analyze all results in a directory"""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Directory not found: {results_dir}")
        return

    json_files = list(results_path.glob("*.json"))
    if not json_files:
        print(f"No JSON result files found in {results_dir}")
        return

    print(f"\nANALYZING {len(json_files)} RESULT FILES")
    print("=" * 50)

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if 'personality_interaction' in data:
                # Single negotiation result
                print(f"\n{json_file.name}:")
                print(f"  Personalities: {data.get('personality_interaction', 'unknown')}")
                print(f"  Agreement Ratio: {data.get('outcomes', {}).get('agreement_ratio', 0):.2f}")
                print(f"  Rounds: {data.get('total_rounds', 0)}")
                print(f"  Status: {data.get('status', 'unknown')}")
            else:
                # Batch result
                print(f"\n{json_file.name}: Batch result with {len(data)} tests")

        except Exception as e:
            print(f"\nError reading {json_file.name}: {e}")


if __name__ == "__main__":
    asyncio.run(main())