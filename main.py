from util import agent, routines, tools


class Bot(agent.VirxERLU):
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
        if not self.kickoff_done:
            # Kickoff code
            if self.is_clear():
                self.push(routines.generic_kickoff())
        elif self.is_clear():
            if self.me.boost > 36:
                # If we have more than 36 boost, then look for a shot on the opp's net

                goal = (self.foe_goal.left_post, self.foe_goal.right_post)
                jump_shot = tools.find_jump_shot(self, goal)
                aerial_shot = tools.find_aerial(self, goal)
                shots = []

                if jump_shot is not None:
                    shots.append(jump_shot)

                if aerial_shot is not None:
                    shots.append(aerial_shot)

                if len(shots) == 2:
                    # If we found 2 shots, sort them by how fast they are
                    shots.sort(key=lambda shot: shot.intercept_time)

                # If we couldn't find a shot, push a short shot. Otherwise, go for the fastest shot.
                self.push(routines.short_shot(self.foe_goal.location) if len(shots) == 0 else shots[0])
            else:
                # If we don't have enough boost, go for more boost
                boosts = list(boost for boost in self.boosts if boost.active and boost.large)
                boosts.sort(key=lambda boost: self.me.location.flat_dist(boost.location))
                self.push(routines.goto_boost(boosts[0]))

    def handle_match_comm(self, msg):
        # This is for handling any incoming match communications
        self.print(msg)

    def handle_quick_chat(self, index, team, quick_chat):
        # This is for hanling any incoming quick chats
        if self.team is team:
            self.print(quick_chat)
