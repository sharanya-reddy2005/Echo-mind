import os, json, csv, datetime
import networkx as nx

UTILS_DIR = "exports"
os.makedirs(UTILS_DIR, exist_ok=True)


class Utils:
    # -------------------------
    # Save JSON
    # -------------------------
    @staticmethod
    def save_json(data, filename):
        path = os.path.join(UTILS_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    # -------------------------
    # Save CSV
    # -------------------------
    @staticmethod
    def save_csv(data, filename, headers=None):
        path = os.path.join(UTILS_DIR, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if headers:
                writer.writerow(headers)
            for row in data:
                writer.writerow(row)
        return path

    # -------------------------
    # Export Debate Graph to GraphML (for Gephi/Neo4j)
    # -------------------------
    @staticmethod
    def export_graphml(transcript, filename="debate.graphml"):
        G = nx.DiGraph()
        for i in range(1, len(transcript)):
            prev_agent = transcript[i - 1]["agent"]
            curr_agent = transcript[i]["agent"]
            G.add_edge(prev_agent, curr_agent, text=transcript[i]["text"])

        path = os.path.join(UTILS_DIR, filename)
        nx.write_graphml(G, path)
        return path

    # -------------------------
    # Session Logger
    # -------------------------
    @staticmethod
    def log_session(topic, agents, transcript, votes, decision):
        log = {
            "topic": topic,
            "agents": [a.name for a in agents],
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "transcript_len": len(transcript),
            "votes": votes,
            "final_decision": decision
        }
        filename = f"log_{int(datetime.datetime.utcnow().timestamp())}.json"
        return Utils.save_json(log, filename)

    # -------------------------
    # Pretty Print for Debugging
    # -------------------------
    @staticmethod
    def pretty_print_transcript(transcript):
        print("\n--- Debate Transcript ---")
        for t in transcript:
            print(f"[{t['agent']}] {t['text']}")

    @staticmethod
    def pretty_print_votes(votes):
        print("\n--- Votes ---")
        for v in votes:
            print(f"{v['agent']} → {v['vote']} | {v['text']}")
