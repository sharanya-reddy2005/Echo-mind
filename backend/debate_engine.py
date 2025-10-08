class DebateEngine:
    def __init__(self, agents, client, rounds=3, memory_enabled=True):
        self.agents = agents
        self.client = client
        self.rounds = rounds
        self.memory_enabled = memory_enabled

    def run(self, topic):
        """Standard debate run (returns full transcript at once)."""
        transcript = []
        for r in range(self.rounds):
            for agent in self.agents:
                text = agent.speak(self.client, topic, transcript, memory_enabled=self.memory_enabled)
                turn = {"agent": agent.name, "role": agent.role, "text": text, "round": r+1}
                transcript.append(turn)
        return transcript

    def run_streaming(self, topic):
        """Streaming debate run (yields each agent's turn dynamically)."""
        transcript = []
        for r in range(self.rounds):
            for agent in self.agents:
                # Each agent speaks
                text = agent.speak(self.client, topic, transcript, memory_enabled=self.memory_enabled)

                # Build turn record
                turn = {
                    "agent": agent.name,
                    "role": agent.role,
                    "text": text,
                    "round": r + 1
                }
                transcript.append(turn)

                # Yield this turn immediately
                yield turn
