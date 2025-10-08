import dearpygui.dearpygui as dpg
from backend.app import run_debate   # import your backend debate runner

API_KEY = "CFM82TPxkuGTDgnljYHZNLM7A50FE9XP"

def start_debate_callback(sender, app_data, user_data):
    topic = dpg.get_value("topic_input")
    num_agents = dpg.get_value("agent_slider")
    rounds = dpg.get_value("rounds_slider")

    dpg.set_value("status_text", f"⚡ Running debate on: {topic}...")
    dpg.configure_item("results_window", show=True)

    try:
        result = run_debate(API_KEY, topic, num_agents, rounds)

        # Clear old logs
        dpg.delete_item("transcript_child", children_only=True)

        # Show transcript
        for turn in result["transcript"]:
            dpg.add_text(f"{turn['agent']}: {turn['text'][:150]}...", parent="transcript_child", wrap=600)

        # Show votes
        dpg.delete_item("votes_child", children_only=True)
        for v in result["votes"]:
            dpg.add_text(f"{v['agent']} → {v['vote']}: {v['text'][:120]}...", parent="votes_child", wrap=600)

        # Show final decision
        dpg.set_value("winner_text", f"🏆 Final Decision: {result['final_decision']}")

    except Exception as e:
        dpg.set_value("status_text", f"❌ Error: {e}")

# -----------------------------
# Main UI
# -----------------------------
dpg.create_context()
dpg.create_viewport(title="⚖️ EchoMind — Debate Arena", width=1000, height=700)

with dpg.window(label="Debate Arena", width=980, height=680):
    dpg.add_input_text(label="Debate Topic", tag="topic_input", hint="Enter your topic here")
    dpg.add_slider_int(label="Agents", tag="agent_slider", min_value=2, max_value=8, default_value=4)
    dpg.add_slider_int(label="Rounds", tag="rounds_slider", min_value=1, max_value=10, default_value=3)
    dpg.add_button(label="🚀 Start Debate", callback=start_debate_callback)
    dpg.add_text("", tag="status_text")

    with dpg.collapsing_header(label="Debate Results", tag="results_window", default_open=True, show=False):
        dpg.add_text("Transcript", bullet=True)
        dpg.add_child_window(tag="transcript_child", autosize_x=True, height=200)

        dpg.add_text("Votes", bullet=True)
        dpg.add_child_window(tag="votes_child", autosize_x=True, height=150)

        dpg.add_separator()
        dpg.add_text("", tag="winner_text", color=[0, 200, 0])

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
