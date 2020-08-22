from util import routines, tools, utils
from util.agent import VirxERLU, Vector


class Bot(VirxERLU):
    # If your bot encounters an error, VirxERLU will do it's best to keep your bot from crashing.
    # VirxERLU uses a stack system for it's routines. A stack is a first-in, last-out system. The stack is a list of routines.
    # VirxERLU on VirxEC Showcase -> https://virxerlu.virxcase.dev/
    # Wiki -> https://github.com/VirxEC/VirxERLU/wiki
    def init(self):
        # Put any initialization code here
        pass

    def run(self):
        # If the stack is clear
        if self.is_clear():
            # push our atba (at the ball agent) routine to the stack
            self.push(routines.atba())

    def demolished(self):
        # If the stack isn't clear
        if not self.is_clear():
            # Clear the stack
            self.clear()

    def handle_match_comm(self, msg):
        # This is for handling any incoming match communications
        # All match comms are Python objects
        if msg.get('team') is self.team:
            self.print(msg)

    def handle_quick_chat(self, index, team, quick_chat):
        # This is for handling any incoming quick chats
        # See https://github.com/RLBot/RLBot/blob/master/src/main/flatbuffers/rlbot.fbs#L376 for a list of all quick chats
        if self.team is team:
            self.print(quick_chat)
