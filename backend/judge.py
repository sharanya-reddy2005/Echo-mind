import re

class JudgeAgent:
    def __init__(self, client, model="mistral-tiny"):
        self.client = client
        self.model = model

    # -------------------------
    # Baseline: LLM-as-a-Judge
    # -------------------------
    def make_decision(self, topic, transcript, votes):
        transcript_text = "\n".join([f"{t['agent']}: {t['text']}" for t in transcript[-30:]])
        votes_text = "\n".join([f"{v['agent']} voted {v['vote']}" for v in votes])

        system_msg = f"You are the impartial Judge of a debate on '{topic}'. Evaluate transcript + votes and declare FINAL DECISION."
        user_msg = f"Transcript:\n{transcript_text}\n\nVotes:\n{votes_text}\n\nIssue your ruling."

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()

    # -------------------------
    # ChatEval: Multi-Agent Judges
    # -------------------------
    def chateval_decision(self, topic, transcript, votes, num_judges=3):
        transcript_text = "\n".join([f"{t['agent']}: {t['text']}" for t in transcript[-30:]])
        votes_text = "\n".join([f"{v['agent']} voted {v['vote']}" for v in votes])

        decisions = []
        for i in range(num_judges):
            system_msg = f"You are Judge #{i+1} in a debate on '{topic}'. Evaluate and issue FINAL DECISION."
            user_msg = f"Transcript:\n{transcript_text}\n\nVotes:\n{votes_text}\n\nIssue your ruling."
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                max_tokens=200
            )
            decision_text = resp.choices[0].message.content.strip()
            match = re.search(r"FINAL DECISION:\s*(\w+)", decision_text)
            final = match.group(1) if match else decision_text
            decisions.append(final)

        # Aggregate decision (majority among judges)
        from collections import Counter
        tally = Counter(decisions)
        return tally.most_common(1)[0][0]

    # -------------------------
    # Cross-Judging: Agents critique Judge
    # -------------------------
    def cross_judging(self, client, agents, topic, judge_decision, transcript):
        critiques = []
        for agent in agents:
            prompt = f"""
            Debate Topic: '{topic}'
            Judge's Decision: {judge_decision}
            Transcript: {transcript[-10:]}
            
            As {agent.name}, critique the Judge's ruling.
            Do you agree or disagree? Provide 2-3 sentences.
            """
            resp = client.chat.completions.create(
                model=agent.model,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=120
            )
            critique = resp.choices[0].message.content.strip()
            critiques.append({"agent": agent.name, "critique": critique})
        return critiques

    # -------------------------
    # Adversarial Adjudication
    # -------------------------
    def adversarial_decision(self, topic, transcript, votes):
        transcript_text = "\n".join([f"{t['agent']}: {t['text']}" for t in transcript[-30:]])
        votes_text = "\n".join([f"{v['agent']} voted {v['vote']}" for v in votes])

        system_msg = f"""
        You are the Judge of a debate on '{topic}'.
        Task: Identify the most factually accurate AND logically strong argument.
        - Do NOT reward persuasion tricks or emotional appeals.
        - Prioritize factual accuracy, consistency, and reasoning.
        - Final output must follow format:
        FINAL DECISION: Agent_X
        Justification: <3-5 sentences>
        """

        user_msg = f"Transcript:\n{transcript_text}\n\nVotes:\n{votes_text}\n\nNow issue your ruling."

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()
