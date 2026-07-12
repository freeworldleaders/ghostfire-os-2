from agents.base import Agent

class AgentRegistry:
    def __init__(self):
        self.agents = {}

    def register(self, name):
        self.agents[name] = Agent(name)

    def start_all(self):
        for agent in self.agents.values():
            agent.run()
