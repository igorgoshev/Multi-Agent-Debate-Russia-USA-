#!/usr/bin/env python3
"""
Easy Configuration System for AI Negotiation
Customizable settings for personalities, topics, and parameters

Usage:
python config_negotiation.py --help
python config_negotiation.py --scenario climate --usa hawk --russia dove
"""

import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

@dataclass
class NegotiationConfig:
    """Complete negotiation configuration"""
    # Scenario settings
    scenario_name: str = "ukraine"
    topic: str = "Ukraine conflict resolution"
    initial_prompt: str = ""
    context_queries: List[str] = None
    
    # Agent settings
    usa_personality: str = "dove"
    russia_personality: str = "hawk"
    usa_model: str = "qwen3:8b"
    russia_model: str = "llama3:8b"
    
    # Negotiation parameters
    max_rounds: int = 8
    phase_transition_threshold: int = 2
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # RAG parameters
    rag_top_k: int = 3
    rag_score_threshold: float = 0.6
    max_context_length: int = 2000
    
    # Monitoring
    enable_monitoring: bool = False
    enable_web_monitor: bool = False
    save_detailed_logs: bool = True
    
    def __post_init__(self):
        if self.context_queries is None:
            self.context_queries = []

# Personality definitions
PERSONALITIES = {
    "hawk": {
        "description": """
        You are a security-focused hardliner who prioritizes national interests above all else.
        
        CORE TRAITS:
        - Take strong initial positions and defend them vigorously
        - Demand significant concessions before offering any
        - Use historical precedents to justify tough stances
        - View negotiations as zero-sum competitions
        - Show minimal flexibility on core security issues
        
        COMMUNICATION STYLE:
        - Direct and assertive language
        - Reference military capabilities and deterrence
        - Emphasize historical grievances and threats
        - Frame issues in terms of national security
        - Use conditional language for any concessions
        """,
        "traits": ["assertive", "security_focused", "history_oriented", "minimal_compromise"]
    },
    
    "dove": {
        "description": """
        You are a diplomatic mediator focused on finding mutually beneficial solutions.
        
        CORE TRAITS:
        - Seek common ground and shared interests early
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
        """,
        "traits": ["collaborative", "flexible", "relationship_focused", "compromise_willing"]
    },
    
    "economist": {
        "description": """
        You prioritize economic benefits and practical trade relationships.
        
        CORE TRAITS:
        - Focus negotiations on economic impacts and trade flows
        - Willing to compromise on political issues for economic gains
        - Use market data and economic analysis in arguments
        - Prefer phased implementation with economic incentives
        - Show flexibility when presented with profitable alternatives
        
        COMMUNICATION STYLE:
        - Data-driven and analytical language
        - Reference trade statistics and economic projections
        - Emphasize market opportunities and growth potential
        - Propose economic incentives and business partnerships
        - Frame agreements in terms of economic competitiveness
        """,
        "traits": ["data_driven", "pragmatic", "trade_focused", "incentive_oriented"]
    },
    
    "legalist": {
        "description": """
        You emphasize international law, treaties, and established precedents.
        
        CORE TRAITS:
        - Reference specific treaties and legal frameworks frequently
        - Insist on legally binding agreements with enforcement mechanisms
        - Use international court decisions to support positions
        - Prefer formal diplomatic protocols and procedures
        - Show flexibility only within established legal boundaries
        
        COMMUNICATION STYLE:
        - Formal and precise legal language
        - Quote specific articles and legal precedents
        - Emphasize binding commitments and enforcement
        - Reference international institutions and courts
        - Propose detailed legal frameworks and procedures
        """,
        "traits": ["law_focused", "formal", "precedent_oriented", "binding_agreements"]
    },
    
    "innovator": {
        "description": """
        You focus on creative solutions and new approaches to old problems.
        
        CORE TRAITS:
        - Propose unconventional solutions and creative frameworks
        - Willing to try experimental approaches and pilot programs
        - Focus on technology and innovation as solution enablers
        - Embrace risk-taking for potentially high rewards
        - Show openness to new formats and negotiation structures
        
        COMMUNICATION STYLE:
        - Creative and forward-thinking language
        - Reference emerging technologies and trends
        - Propose experimental and pilot approaches
        - Use metaphors and analogies to explain new concepts
        - Emphasize adaptation and evolution over tradition
        """,
        "traits": ["creative", "experimental", "tech_oriented", "risk_taking"]
    }
}

# Scenario templates
SCENARIOS = {
    "ukraine": {
        "topic": "Ukraine conflict resolution and future security arrangements",
        "initial_prompt": """
        We are here to discuss a potential framework for resolving the ongoing conflict in Ukraine and establishing future security arrangements in Eastern Europe.
        
        Key issues to address:
        1. Territorial integrity and sovereignty principles
        2. Security guarantees and alliance relationships  
        3. Economic reconstruction and cooperation
        4. Humanitarian concerns and refugee support
        5. Future diplomatic engagement mechanisms
        
        Both parties should present their key concerns, priorities, and potential areas for compromise.
        Please begin by stating your primary position and key requirements for any potential agreement.
        """,
        "context_queries": ["Ukraine conflict", "security guarantees", "territorial integrity", "NATO expansion", "economic sanctions"]
    },
    
    "climate": {
        "topic": "International climate cooperation and emissions reduction framework",
        "initial_prompt": """
        Negotiate a comprehensive climate cooperation agreement addressing the urgency of global warming.
        
        Key issues to negotiate:
        1. National emissions reduction targets and timelines
        2. Technology transfer and green innovation sharing
        3. Climate finance commitments and mechanisms
        4. Monitoring, reporting, and verification systems
        5. Support for developing nations and just transition
        
        Balance economic development needs with environmental protection requirements.
        Present your country's climate priorities and acceptable commitments.
        """,
        "context_queries": ["climate change", "emissions targets", "green technology", "climate finance", "environmental policy"]
    },
    
    "trade": {
        "topic": "Bilateral trade agreement and economic partnership",
        "initial_prompt": """
        Establish a comprehensive bilateral trade agreement that benefits both economies.
        
        Key areas for negotiation:
        1. Tariff reductions on key commodities and manufactured goods
        2. Market access for services and digital trade
        3. Intellectual property protections and enforcement
        4. Investment protections and dispute resolution
        5. Labor and environmental standards alignment
        
        Each side should protect key domestic industries while maximizing mutual trade benefits.
        Outline your trade priorities and red lines for the agreement.
        """,
        "context_queries": ["trade statistics", "tariff impacts", "export opportunities", "industry competitiveness", "trade agreements"]
    },
    
    "cyber": {
        "topic": "Cybersecurity cooperation and digital governance framework",
        "initial_prompt": """
        Develop a bilateral cybersecurity cooperation agreement addressing digital threats and governance.
        
        Critical areas to address:
        1. Cybercrime prevention and law enforcement cooperation
        2. Critical infrastructure protection mechanisms
        3. Information sharing protocols and threat intelligence
        4. Digital sovereignty and data governance standards
        5. Norms for state behavior in cyberspace
        
        Balance security cooperation needs with sovereignty and privacy concerns.
        Present your cybersecurity priorities and cooperation boundaries.
        """,
        "context_queries": ["cybersecurity threats", "digital governance", "cyber norms", "data protection", "critical infrastructure"]
    },
    
    "energy": {
        "topic": "Energy security and transition cooperation agreement",
        "initial_prompt": """
        Negotiate an energy cooperation framework addressing security and transition needs.
        
        Key negotiation points:
        1. Energy supply diversification and security guarantees
        2. Renewable energy technology cooperation and investment
        3. Traditional energy resource management and pricing
        4. Grid integration and infrastructure development
        5. Just transition support for energy-dependent communities
        
        Balance immediate energy security with long-term transition goals.
        Outline your energy priorities and partnership opportunities.
        """,
        "context_queries": ["energy security", "renewable energy", "energy transition", "pipeline infrastructure", "energy markets"]
    },
    
    "space": {
        "topic": "Space cooperation and governance framework",
        "initial_prompt": """
        Establish a space cooperation agreement covering civilian and security dimensions.
        
        Areas for cooperation:
        1. Peaceful space exploration and scientific collaboration
        2. Space debris mitigation and collision avoidance
        3. Commercial space activities regulation and coordination
        4. Space security and anti-satellite weapon limitations
        5. Moon and asteroid resource utilization frameworks
        
        Promote peaceful space use while addressing security and commercial interests.
        Present your space program priorities and cooperation parameters.
        """,
        "context_queries": ["space cooperation", "satellite technology", "space debris", "space security", "commercial space"]
    }
}

class ConfigurationManager:
    """Manages negotiation configurations and templates"""
    
    def __init__(self):
        self.personalities = PERSONALITIES
        self.scenarios = SCENARIOS
    
    def create_config(
        self,
        scenario: str = "ukraine",
        usa_personality: str = "dove", 
        russia_personality: str = "hawk",
        usa_model: str = "qwen3:8b",
        russia_model: str = "llama3:8b",
        **kwargs
    ) -> NegotiationConfig:
        """Create a negotiation configuration"""
        
        if scenario not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(self.scenarios.keys())}")
        
        if usa_personality not in self.personalities:
            raise ValueError(f"Unknown personality: {usa_personality}. Available: {list(self.personalities.keys())}")
            
        if russia_personality not in self.personalities:
            raise ValueError(f"Unknown personality: {russia_personality}. Available: {list(self.personalities.keys())}")
        
        scenario_data = self.scenarios[scenario]
        
        config = NegotiationConfig(
            scenario_name=scenario,
            topic=scenario_data["topic"],
            initial_prompt=scenario_data["initial_prompt"],
            context_queries=scenario_data["context_queries"],
            usa_personality=usa_personality,
            russia_personality=russia_personality,
            usa_model=usa_model,
            russia_model=russia_model,
            **kwargs
        )
        
        return config
    
    def get_personality_description(self, personality: str) -> str:
        """Get full personality description"""
        if personality not in self.personalities:
            raise ValueError(f"Unknown personality: {personality}")
        return self.personalities[personality]["description"]
    
    def list_available_options(self):
        """Print all available configuration options"""
        print("🎭 AVAILABLE PERSONALITIES:")
        for name, data in self.personalities.items():
            traits = ", ".join(data["traits"])
            print(f"  {name}: {traits}")
        
        print(f"\n📝 AVAILABLE SCENARIOS:")
        for name, data in self.scenarios.items():
            print(f"  {name}: {data['topic']}")
        
        print(f"\n🤖 RECOMMENDED MODELS:")
        models = ["qwen3:8b", "llama3:8b", "mistral:7b", "qwen3:4b", "llama3.2:3b"]
        for model in models:
            print(f"  {model}")
    
    def save_config(self, config: NegotiationConfig, filename: str):
        """Save configuration to JSON file"""
        with open(filename, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        print(f"💾 Configuration saved to {filename}")
    
    def load_config(self, filename: str) -> NegotiationConfig:
        """Load configuration from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return NegotiationConfig(**data)
    
    def create_custom_scenario(
        self,
        name: str,
        topic: str,
        initial_prompt: str,
        context_queries: List[str]
    ):
        """Add a custom scenario"""
        self.scenarios[name] = {
            "topic": topic,
            "initial_prompt": initial_prompt,
            "context_queries": context_queries
        }
        print(f"✅ Custom scenario '{name}' added")

def main():
    """Command line interface for configuration"""
    parser = argparse.ArgumentParser(description="AI Negotiation Configuration System")
    
    # Action selection
    parser.add_argument("action", nargs="?", default="run", 
                       choices=["run", "list", "save-config", "load-config"],
                       help="Action to perform")
    
    # Scenario and personality selection
    parser.add_argument("--scenario", default="ukraine",
                       help="Negotiation scenario")
    parser.add_argument("--usa", dest="usa_personality", default="dove",
                       help="USA agent personality")
    parser.add_argument("--russia", dest="russia_personality", default="hawk", 
                       help="Russia agent personality")
    
    # Model selection
    parser.add_argument("--usa-model", default="qwen3:8b",
                       help="Model for USA agent")
    parser.add_argument("--russia-model", default="llama3.3:8b",
                       help="Model for Russia agent")
    
    # Negotiation parameters
    parser.add_argument("--rounds", type=int, default=8,
                       help="Maximum negotiation rounds")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Model creativity (0.1-1.0)")
    parser.add_argument("--top-k", type=int, default=3,
                       help="Number of context chunks to retrieve")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="RAG similarity threshold")
    
    # Monitoring options
    parser.add_argument("--monitor", action="store_true",
                       help="Enable real-time monitoring")
    parser.add_argument("--web-monitor", action="store_true",
                       help="Enable web dashboard")
    parser.add_argument("--detailed-logs", action="store_true", default=True,
                       help="Save detailed negotiation logs")
    
    # Configuration file options
    parser.add_argument("--config-file", 
                       help="Save/load configuration from file")
    
    # Custom scenario options
    parser.add_argument("--custom-topic",
                       help="Custom negotiation topic")
    parser.add_argument("--custom-prompt",
                       help="Custom initial prompt")
    
    args = parser.parse_args()
    
    config_manager = ConfigurationManager()
    
    if args.action == "list":
        config_manager.list_available_options()
        return
    
    if args.action == "load-config":
        if not args.config_file:
            print("❌ Please specify --config-file for load-config action")
            return
        
        try:
            config = config_manager.load_config(args.config_file)
            print(f"✅ Loaded configuration from {args.config_file}")
            print_config_summary(config)
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
        return
    
    # Create configuration
    try:
        config = config_manager.create_config(
            scenario=args.scenario,
            usa_personality=args.usa_personality,
            russia_personality=args.russia_personality,
            usa_model=args.usa_model,
            russia_model=args.russia_model,
            max_rounds=args.rounds,
            temperature=args.temperature,
            rag_top_k=args.top_k,
            rag_score_threshold=args.threshold,
            enable_monitoring=args.monitor,
            enable_web_monitor=args.web_monitor,
            save_detailed_logs=args.detailed_logs
        )
        
        # Handle custom scenario
        if args.custom_topic:
            config.topic = args.custom_topic
            config.scenario_name = "custom"
            
        if args.custom_prompt:
            config.initial_prompt = args.custom_prompt
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nUse 'python config_negotiation.py list' to see available options")
        return
    
    if args.action == "save-config":
        if not args.config_file:
            filename = f"config_{config.scenario_name}_{config.usa_personality}_vs_{config.russia_personality}.json"
        else:
            filename = args.config_file
            
        config_manager.save_config(config, filename)
        print_config_summary(config)
        return
    
    # Default action: run negotiation
    print("🚀 Starting AI Negotiation with Configuration:")
    print_config_summary(config)
    
    # Import and run negotiation system
    try:
        import asyncio
        from negotiation_agents import run_configured_negotiation
        
        asyncio.run(run_configured_negotiation(config))
        
    except ImportError:
        print("❌ negotiation_agents.py not found. Make sure it's in the same directory.")
        print("💾 Configuration saved for manual use:")
        config_manager.save_config(config, "current_config.json")

def print_config_summary(config: NegotiationConfig):
    """Print a nice summary of the configuration"""
    print("\n" + "="*60)
    print("📊 NEGOTIATION CONFIGURATION SUMMARY")
    print("="*60)
    
    print(f"🎯 Scenario: {config.scenario_name}")
    print(f"📝 Topic: {config.topic}")
    
    print(f"\n🤖 Agents:")
    print(f"   USA: {config.usa_personality} personality, {config.usa_model} model")
    print(f"   Russia: {config.russia_personality} personality, {config.russia_model} model")
    
    print(f"\n⚙️ Parameters:")
    print(f"   Rounds: {config.max_rounds}")
    print(f"   Temperature: {config.temperature}")
    print(f"   RAG top-k: {config.rag_top_k}")
    print(f"   RAG threshold: {config.rag_score_threshold}")
    
    print(f"\n📊 Monitoring:")
    print(f"   Real-time: {'✅' if config.enable_monitoring else '❌'}")
    print(f"   Web dashboard: {'✅' if config.enable_web_monitor else '❌'}")
    print(f"   Detailed logs: {'✅' if config.save_detailed_logs else '❌'}")
    
    print("="*60)

# Integration function for the main negotiation system
async def run_configured_negotiation(config: NegotiationConfig):
    """Run negotiation with the provided configuration"""
    # This would be called from the main negotiation system
    # Import the actual negotiation functions
    
    print(f"🎭 Setting up {config.usa_personality} USA agent vs {config.russia_personality} Russia agent")
    print(f"🎯 Topic: {config.topic}")
    
    # Here you would integrate with your existing negotiation_agents.py
    # by passing the config parameters to the setup functions
    
    # Example integration:
    """
    from negotiation_agents import setup_system, NegotiationManager
    
    # Setup with config parameters
    agents = await setup_system(
        usa_model=config.usa_model,
        russia_model=config.russia_model,
        usa_personality=config.usa_personality,
        russia_personality=config.russia_personality,
        config=config
    )
    
    if agents:
        agent_usa, agent_russia, rag_system = agents
        
        manager = NegotiationManager(agent_usa, agent_russia, config.topic)
        
        results = await manager.run_full_negotiation(
            initial_prompt=config.initial_prompt,
            max_rounds=config.max_rounds,
            phase_transition_threshold=config.phase_transition_threshold
        )
        
        return results
    """
    
    print("⚠️ This is a configuration preview. Integrate with negotiation_agents.py to run actual negotiations.")

# Preset configurations for quick use
PRESET_CONFIGS = {
    "diplomatic": {
        "scenario": "ukraine",
        "usa_personality": "dove", 
        "russia_personality": "dove",
        "description": "Both sides focused on diplomatic solutions"
    },
    
    "hardball": {
        "scenario": "ukraine",
        "usa_personality": "hawk",
        "russia_personality": "hawk", 
        "description": "Tough negotiation with minimal compromise"
    },
    
    "economic": {
        "scenario": "trade",
        "usa_personality": "economist",
        "russia_personality": "economist",
        "description": "Business-focused trade negotiation"
    },
    
    "climate_coop": {
        "scenario": "climate",
        "usa_personality": "innovator",
        "russia_personality": "economist",
        "description": "Innovation vs economic pragmatism on climate"
    },
    
    "mixed": {
        "scenario": "ukraine", 
        "usa_personality": "legalist",
        "russia_personality": "hawk",
        "description": "Law-focused USA vs security-focused Russia"
    }
}

def create_preset_config(preset_name: str) -> NegotiationConfig:
    """Create a preset configuration"""
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    
    preset = PRESET_CONFIGS[preset_name]
    config_manager = ConfigurationManager()
    
    return config_manager.create_config(
        scenario=preset["scenario"],
        usa_personality=preset["usa_personality"],
        russia_personality=preset["russia_personality"]
    )

def show_presets():
    """Show all available preset configurations"""
    print("🎯 AVAILABLE PRESET CONFIGURATIONS:")
    print("="*50)
    
    for name, preset in PRESET_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Description: {preset['description']}")
        print(f"  Scenario: {preset['scenario']}")
        print(f"  USA: {preset['usa_personality']}")
        print(f"  Russia: {preset['russia_personality']}")
        print(f"  Usage: python config_negotiation.py --preset {name}")

if __name__ == "__main__":
    import sys
    
    # Handle preset configurations
    if len(sys.argv) > 1 and sys.argv[1] == "--preset":
        if len(sys.argv) < 3:
            show_presets()
        else:
            preset_name = sys.argv[2]
            try:
                config = create_preset_config(preset_name)
                print_config_summary(config)
                
                # Save preset config
                filename = f"preset_{preset_name}.json"
                ConfigurationManager().save_config(config, filename)
                
            except ValueError as e:
                print(f"❌ {e}")
                show_presets()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--presets":
        show_presets()
    
    else:
        main()