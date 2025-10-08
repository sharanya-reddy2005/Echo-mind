from fastapi import FastAPI
from backend.agent import Agent
from backend.debate_engine import DebateEngine
from backend.consensus import ConsensusEngine
from backend.judge import JudgeAgent

from openai import OpenAI
import datetime

app = FastAPI(title="EchoMind Backend API")


@app.get("/")
def root():
    return {"status": "EchoMind Backend is running"}


@app.post("/debate")
def run_debate(api_key: str, topic: str, num_agents: int = 3, rounds: int = 3):
    client = OpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")

    # Create default agents
    agents = [Agent(name=f"Agent_{i+1}", role="Civilian") for i in range(num_agents)]
    debate = DebateEngine(agents, client, rounds=rounds)
    transcript = debate.run(topic)

    # Voting
    consensus = ConsensusEngine(agents)
    votes = consensus.collect_votes(client, topic, transcript)
    result = consensus.resolve(votes)

    if result == "No Clear Winner":
        judge = JudgeAgent(client)
        result = judge.adversarial_decision(topic, transcript, votes)

    # Return JSON response (no Streamlit here)
    return {
        "topic": topic,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "transcript": transcript,
        "votes": votes,
        "final_decision": result
    }
