from util import routines, tools, utils
from util.agent import VirxERLU, Vector


class Bot(VirxERLU):
    # If your bot encounters an error, VirxERLU will do it's best to keep your bot from crashing.
    # VirxERLU uses a stack system for it's routines. A stack is a first-in, last-out system. The stack is a list of routines.
    def init(self):
        # Any initalization code
        pass

    def demolished(self):
        # Code to run when the bot is demolished
        if not self.is_clear():
            self.clear()

    def run(self):
        # Strategy code
        if self.is_clear():
            self.push(routines.atba())

    def handle_match_comm(self, msg):
        # This is for handling any incoming match communications
        if msg['team'] is self.team:
            self.print(msg)

    def handle_quick_chat(self, index, team, quick_chat):
        # This is for hanling any incoming quick chats
        if self.team is team:
            self.print(quick_chat)
