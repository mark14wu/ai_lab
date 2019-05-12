from copy import deepcopy


class Node:

    # visited = {}

    def __init__(self, state=None, parent=None, cost=0, depth=0, children=[]):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.depth = depth
        self.children = children

    def is_goal(self, goal_state):
        for i in range(len(self.state)):
            for j in range(len(self.state)):
                if self.state[i][j] != goal_state[i][j]:
                    return False
        return True

    def expand(self):
        def next_states(state):
            states = []

            zero_i = None
            zero_j = None

            for i in range(len(state)):
                for j in range(len(state)):
                    if state[i][j] == 0:
                        zero_i = i
                        zero_j = j
                        break

            def add_swap(i, j):
                new_state = deepcopy(state)
                new_state[i][j], new_state[zero_i][zero_j] = new_state[zero_i][zero_j], new_state[i][j]
                states.append(new_state)
                # hashable_state = tuple(tuple(row) for row in new_state)
                # if hashable_state not in Node.visited:
                #     Node.visited[hashable_state] = 0
                #     states.append(new_state)
                #     print("466")
                # else:
                #     print("233")

            if zero_i != 0:
                add_swap(zero_i - 1, zero_j)

            if zero_j != 0:
                add_swap(zero_i, zero_j - 1)

            if zero_i != len(state) - 1:
                add_swap(zero_i + 1, zero_j)

            if zero_j != len(state) - 1:
                add_swap(zero_i, zero_j + 1)

            mid = int(len(state) / 2)

            if zero_i == 0 and zero_j == mid:
                add_swap(len(state) - 1, mid)

            if zero_i == len(state) - 1 and zero_j == mid:
                add_swap(0, mid)

            if zero_i == mid and zero_j == 0:
                add_swap(mid, len(state) - 1)

            if zero_i == mid and zero_j == len(state) - 1:
                add_swap(mid, 0)

            return states

        new_states = next_states(self.state)
        self.children = []
        for state in new_states:
            self.children.append(Node(state, self, self.cost + 1, self.depth + 1))

    def parents(self):
        current_node = self
        while current_node.parent:
            yield current_node.parent
            current_node = current_node.parent

    def gn(self):
        costs = self.cost
        for parent in self.parents():
            costs += parent.cost
        return costs
