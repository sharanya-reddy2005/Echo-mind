import json
from backend.memory import MemoryManager
# import time
# import openai

# def safe_completion(client, **kwargs):
#     """Call OpenAI with retries on rate-limit errors."""
#     for attempt in range(5):
#         try:
#             return client.chat.completions.create(**kwargs)
#         except openai.RateLimitError as e:
#             wait = 2 ** attempt
#             print(f"Rate limit hit. Retrying in {wait}s...")
#             time.sleep(wait)
#         except Exception as e:
#             raise e
#     raise RuntimeError("Failed after retries due to repeated rate-limit errors.")

ROLE_CONFIG = {
    "Leader": {"self": 0.6, "others": 0.4},
    "Architect": {"self": 0.5, "others": 0.5},
    "Contrarian": {"self": 0.7, "others": 0.3},
    "Validator": {"self": 0.4, "others": 0.6},
    "Harmonizer": {"self": 0.3, "others": 0.7},
    "Visionary": {"self": 0.5, "others": 0.5},
    "Strategist": {"self": 0.55, "others": 0.45},
    "Civilian": {"self": 0.3, "others": 0.7},
    "Rebel": {"self": 0.8, "others": 0.2},
    "Judge": {"self": 1.0, "others": 0.0}
}


class Agent:
    def __init__(self, name, role, stance="neutral", persona="factual", model="mistral-small"):
        self.name = name
        self.role = role
        self.stance = stance
        self.persona = persona
        self.model = model
        self.history = []   # short-term (within debate)
        self.memory = []    # long-term (across debates)
        self.memory_manager = MemoryManager(use_vector_db=False)  # optional vector DB later

    def add_memory(self, text: str):
        """Store experience in both short-term and long-term memory."""
        self.history.append(text)
        self.memory.append(text)
        self.memory_manager.save(self.name, text)

    def recall(self, query: str, top_k=3):
        """Retrieve relevant long-term memories."""
        return self.memory_manager.retrieve(self.name, query, top_k=top_k)

    def speak(self, client, topic, transcript, memory_enabled=True):
        """
        Generate an argument using stance, persona, and role influence.
        """
        # Role weighting
        weights = ROLE_CONFIG.get(self.role, {"self": 0.5, "others": 0.5})

        # Transcript context (last 5 turns)
        others_context = "\n".join(
            [f"{t['agent']} ({t['role']}): {t['text']}" for t in transcript[-5:]]
        )
        self_context = f"My past arguments: {self.history[-2:]}" if self.history else ""

        # Add memory context if enabled
        mem_context = ""
        if memory_enabled:
            recalled = self.recall(topic, top_k=2)
            if recalled:
                mem_context = f"Relevant memories: {json.dumps(recalled, indent=2)}"

        # Build debate prompt
        prompt = f"""
        You are {self.name}, role: {self.role}, stance: {self.stance}, persona: {self.persona}.
        Debate Topic: {topic}

        {weights['self']:.0%} weight: Stick to your own stance and persona.
        {weights['others']:.0%} weight: Respond to recent arguments from others.

        Debate so far:
        {others_context}

        {self_context}
        {mem_context}

        Respond concisely with your next argument, keeping role and persona in mind.
        """

        # Call LLM API
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI debate agent."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        text = resp.choices[0].message.content.strip()

        # Save to memory
        self.add_memory(text)

        return text
