import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from collections import defaultdict, Counter


class AnalysisTools:
    # -------------------------
    # Vote Distribution
    # -------------------------
    @staticmethod
    def plot_vote_distribution(votes):
        counts = Counter([v["vote"] for v in votes])
        fig, ax = plt.subplots()
        ax.bar(list(counts.keys()), list(counts.values()), color="skyblue")
        ax.set_title("Vote Distribution")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # -------------------------
    # Debate Flow Graph
    # -------------------------
    @staticmethod
    def plot_influence_graph(transcript):
        G = nx.DiGraph()

        for i in range(1, len(transcript)):
            prev_agent = transcript[i - 1]["agent"]
            curr_agent = transcript[i]["agent"]
            G.add_edge(prev_agent, curr_agent)

        fig, ax = plt.subplots(figsize=(6, 4))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1500, font_size=10, ax=ax, edge_color="gray")
        st.pyplot(fig)

    # -------------------------
    # Coalition Dynamics
    # -------------------------
    @staticmethod
    def analyze_coalitions(votes):
        coalitions = defaultdict(list)
        for v in votes:
            coalitions[v["vote"]].append(v["agent"])

        st.subheader("🤝 Coalition Dynamics")
        for choice, members in coalitions.items():
            st.write(f"**{choice}**: {', '.join(members)}")

        # Visualize as bipartite graph
        G = nx.Graph()
        for choice, members in coalitions.items():
            for m in members:
                G.add_edge(m, choice)

        fig, ax = plt.subplots(figsize=(6, 4))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color="lightgreen", node_size=1200, font_size=10, ax=ax)
        st.pyplot(fig)

    # -------------------------
    # Agent Influence Metrics
    # -------------------------
    @staticmethod
    def agent_influence(transcript):
        G = nx.DiGraph()
        for i in range(1, len(transcript)):
            prev_agent = transcript[i - 1]["agent"]
            curr_agent = transcript[i]["agent"]
            G.add_edge(prev_agent, curr_agent)

        centrality = nx.degree_centrality(G)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

        st.subheader("📊 Agent Influence Metrics")
        for agent, score in sorted_centrality:
            st.write(f"**{agent}**: {score:.2f}")

        # Plot centrality scores
        fig, ax = plt.subplots()
        ax.bar([a for a, _ in sorted_centrality], [s for _, s in sorted_centrality], color="orange")
        ax.set_title("Agent Influence (Centrality)")
        ax.set_ylabel("Score")
        st.pyplot(fig)
