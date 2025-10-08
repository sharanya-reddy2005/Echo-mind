import streamlit as st
import datetime, time, json
from openai import OpenAI

from backend.agent import Agent
from backend.debate_engine import DebateEngine
from backend.consensus import ConsensusEngine
from backend.judge import JudgeAgent
from backend.analysis import AnalysisTools
from backend.utils import Utils

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

ROLE_STYLES = {
    "Architect": ("👑", "#ffb347"),
    "Contrarian": ("⚡", "#ff6b6b"),
    "Harmonizer": ("🛡️", "#4ecdc4"),
    "Validator": ("🔍", "#6a5acd"),
    "Visionary": ("💡", "#ffcc00"),
    "Strategist": ("🎯", "#00b894"),
    "Leader": ("📢", "#0984e3"),
    "Rebel": ("🔥", "#e17055"),
    "Civilian": ("🙂", "#b2bec3"),
}

def render_agent_message(agent_name, role, text):
    icon, color = ROLE_STYLES.get(role, ("💬", "#636e72"))
    return f"""
    <div style="background-color:#1e272e; border-radius:10px; padding:15px; margin:10px 0;
                border-left:6px solid {color}; box-shadow:0px 2px 6px rgba(0,0,0,0.3);">
        <div style="font-weight:bold; color:{color}; font-size:16px; margin-bottom:5px;">
            {icon} {agent_name} • {role}
        </div>
        <div style="color:#dfe6e9; font-size:14px; line-height:1.5;">
            {text}
        </div>
    </div>
    """

def main():
    st.set_page_config(page_title="EchoMind — Debate Arena", layout="wide")
    st.title("⚖️ EchoMind — Emergent Intelligence through Autonomous LLM Societies")

    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        mistral_key = st.text_input("Mistral API Key", type="password")
        base_url = st.text_input("API Base URL", value=MISTRAL_BASE_URL)
        num_agents = st.slider("Number of Agents", 2, 8, 3)
        rounds = st.number_input("Debate Rounds", min_value=1, max_value=10, value=3)
        memory_enabled = st.checkbox("Enable Memory", value=True)

    if not mistral_key:
        st.warning("Enter your Mistral API key in the sidebar.")
        return

    client = OpenAI(api_key=mistral_key, base_url=base_url)
    st.success("✅ Mistral Client Initialized")

    # Setup agents
    agents = []
    for i in range(num_agents):
        with st.sidebar.expander(f"Agent {i+1}"):
            name = st.text_input(f"Name (A{i+1})", value=f"Agent_{i+1}", key=f"name_{i}")
            role = st.selectbox(f"Role (A{i+1})", list(ROLE_STYLES.keys()), key=f"role_{i}")
            stance = st.selectbox(f"Stance (A{i+1})", ["for", "against", "neutral"], key=f"stance_{i}")
            persona = st.selectbox(f"Persona (A{i+1})", ["factual", "funny", "serious"], key=f"persona_{i}")
        agents.append(Agent(name=name, role=role, stance=stance, persona=persona))

    topic = st.text_input("💡 Debate Topic")
    if st.button("🚀 Start Debate"):
        if not topic.strip():
            st.error("Enter a valid debate topic.")
            return

        st.subheader("🗣️ Live Debate Transcript")
        debate = DebateEngine(agents, client, rounds=rounds, memory_enabled=memory_enabled)

        # Dynamic debate streaming
        transcript = []
        transcript_container = st.container()
        for turn in debate.run_streaming(topic):  # <-- use streaming generator
            transcript.append(turn)
            with transcript_container:
                st.markdown(render_agent_message(turn["agent"], turn.get("role", "Civilian"), turn["text"]),
                            unsafe_allow_html=True)
            time.sleep(0.5)  # simulate pacing

        # ----------------------------
        # Voting
        # ----------------------------
        st.subheader("🗳️ Voting Results")
        consensus_engine = ConsensusEngine(agents)
        votes = consensus_engine.collect_votes(client, topic, transcript)

        result = consensus_engine.resolve(votes, method="majority")

        if result == "No Clear Winner":
            judge = JudgeAgent(client)
            result = judge.adversarial_decision(topic, transcript, votes)
            st.warning("⚖️ Judge Decision Invoked")

        # Display winner with role
        winner_role = None
        for a in agents:
            if a.name == result:
                winner_role = a.role
                break
        st.success(f"🏆 Final Decision: {result} ({winner_role})")

        # ----------------------------
        # Save & Download
        # ----------------------------
        output = {
            "topic": topic,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agents": [{ "name": a.name, "role": a.role } for a in agents],
            "transcript": transcript,
            "votes": votes,
            "final_decision": f"{result} ({winner_role})"
        }
        Utils.save_json(output, f"echomind_session_{int(time.time())}.json")
        st.download_button("⬇ Download Transcript", data=json.dumps(output, indent=2), file_name="debate.json")

        # ----------------------------
        # Analysis
        # ----------------------------
        st.subheader("📊 Debate Analysis")
        AnalysisTools.plot_vote_distribution(votes)
        AnalysisTools.plot_influence_graph(transcript)
        AnalysisTools.analyze_coalitions(votes)
        AnalysisTools.agent_influence(transcript)


if __name__ == "__main__":
    main()
