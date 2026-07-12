from runtime.engine import RuntimeEngine
from router.router import CommandRouter

RuntimeEngine().start()
CommandRouter().execute("BOOT")
