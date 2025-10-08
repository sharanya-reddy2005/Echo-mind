from collections import Counter
import numpy as np
import re


class ConsensusEngine:
    def __init__(self, agents):
        self.agents = agents

    # -------------------------
    # Voting Collection
    # -------------------------
    def collect_votes(self, client, topic, transcript):
        """
        Collects initial votes from agents.
        Always sends both system + user messages to avoid empty conversation errors.
        """
        votes = []
        for agent in self.agents:
            prompt = f"""
            Debate on '{topic}' has concluded.

            You are {agent.role} ({agent.name}).
            Review the transcript below and cast your vote:

            Transcript: {transcript}

            Reply in format:

            VOTE: AgentName
            Justification: <short reason>
            """

            try:
                resp = client.chat.completions.create(
                    model=agent.model,
                    messages=[
                        {"role": "system", "content": "You are participating in a consensus voting process."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150
                )
                text = resp.choices[0].message.content.strip()
            except Exception as e:
                text = "VOTE: abstain\nJustification: API error"

            match = re.search(r"VOTE:\s*([\w\d_]+)", text)
            choice = match.group(1) if match else "abstain"
            votes.append({"agent": agent.name, "vote": choice, "text": text})
        return votes

    # -------------------------
    # Majority Voting
    # -------------------------
    def majority_vote(self, votes):
        tally = Counter([v["vote"] for v in votes if v.get("vote")])
        if not tally:
            return "No Decision"
        winner, count = tally.most_common(1)[0]
        # Handle tie
        if list(tally.values()).count(count) > 1:
            return "No Clear Winner"
        return winner

    # -------------------------
    # Borda Count
    # -------------------------
    def borda_count(self, rankings):
        scores = Counter()
        if not rankings:
            return "No Rankings Provided"
        for rank in rankings:
            for i, agent in enumerate(rank):
                scores[agent] += len(rank) - i - 1
        return scores.most_common(1)[0][0] if scores else "No Decision"

    # -------------------------
    # Condorcet Method
    # -------------------------
    def condorcet(self, pairwise_preferences):
        if not pairwise_preferences:
            return "No Pairwise Data"
        candidates = list(pairwise_preferences.keys())
        for c in candidates:
            wins = sum(pairwise_preferences[c][opp] > pairwise_preferences[opp][c]
                       for opp in candidates if opp != c)
            if wins == len(candidates) - 1:
                return c
        return "No Condorcet Winner"

    # -------------------------
    # Deliberation-based Consensus
    # -------------------------
    def deliberation_consensus(self, stance_values, threshold=0.7):
        if not stance_values:
            return "NO CONSENSUS"
        avg = np.mean(stance_values)
        if avg > threshold:
            return "FOR"
        elif avg < -threshold:
            return "AGAINST"
        return "NO CONSENSUS"

    # -------------------------
    # CONSENSAGENT Refinement
    # -------------------------
    def consensagent_refinement(self, client, topic, transcript, raw_votes):
        refined_votes = []
        for agent in self.agents:
            prev_vote = next((v['vote'] for v in raw_votes if v['agent'] == agent.name), 'abstain')
            prompt = f"""
            In the debate on '{topic}', you initially voted: {prev_vote}.

            Reconsider carefully:
            - Do NOT follow majority blindly.
            - Base decision on factual accuracy and reasoning strength.

            Reply again in format:

            VOTE: AgentName
            Justification: <short reason>
            """
            try:
                resp = client.chat.completions.create(
                    model=agent.model,
                    messages=[
                        {"role": "system", "content": "You are refining your vote to reduce sycophancy."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150
                )
                text = resp.choices[0].message.content.strip()
            except Exception:
                text = "VOTE: abstain\nJustification: API error"

            match = re.search(r"VOTE:\s*([\w\d_]+)", text)
            choice = match.group(1) if match else "abstain"
            refined_votes.append({"agent": agent.name, "vote": choice, "text": text})
        return refined_votes

    # -------------------------
    # Resolution Pipeline
    # -------------------------
    def resolve(self, votes, method="majority", **kwargs):
        if not votes:
            return "No Votes"

        if method == "majority":
            return self.majority_vote(votes)

        elif method == "borda":
            rankings = kwargs.get("rankings", [])
            return self.borda_count(rankings)

        elif method == "condorcet":
            pairwise = kwargs.get("pairwise", {})
            return self.condorcet(pairwise)

        elif method == "deliberation":
            stance_values = kwargs.get("stances", [0 for _ in votes])
            return self.deliberation_consensus(stance_values)

        elif method == "consensagent":
            client = kwargs.get("client")
            topic = kwargs.get("topic", "")
            transcript = kwargs.get("transcript", [])
            refined = self.consensagent_refinement(client, topic, transcript, votes)
            return self.majority_vote(refined)

        return "No Consensus Method Selected"
        return final    
