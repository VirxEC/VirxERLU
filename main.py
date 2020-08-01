from util import agent, routines, tools


class Bot(agent.VirxERLU):
    # If your bot encounters an error, VirxERLU will do it's best to keep your bot from crashing.
    # VirxERLU uses a stack system for it's routines. A stack is a first-in, last-out system. The stack is a list of routines.
    def init(self):
        # Any initalization code
        pass

    def demolished(self):
        # Code to run when the bot is demolished
        pass

    def run(self):
        # Strategy code
        pass

    def handle_match_comm(self, msg):
        # This is for handling any incoming match communications
        if msg['team'] is self.team:
            self.print(msg)

    def handle_quick_chat(self, index, team, quick_chat):
        # This is for hanling any incoming quick chats
        if self.team is team:
            self.print(quick_chat)
