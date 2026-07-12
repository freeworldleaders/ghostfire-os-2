from runtime.engine import RuntimeEngine
from router.router import CommandRouter
from agents.registry import AgentRegistry

RuntimeEngine().start()

router = CommandRouter()
router.execute("BOOT")

registry = AgentRegistry()
registry.register("Commander")
registry.register("Guardian")
registry.start_all()
