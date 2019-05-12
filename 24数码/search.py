import heapq
from typing import List, Any

from node import Node


class Search:

    goal_state = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],\
                  [16, 17, 18, 19, 20], [21, 22, 23, 24, 0]]
    tiles_places = []
    for i in range(len(goal_state)):
        for j in range(len(goal_state)):
            heapq.heappush(tiles_places, (goal_state[i][j], (i, j)))

    def get_state():
        state: List[List[int]] = []
        n = int(input())
        for i in range(n):
            row = [int(x) for x in input().split(' ')]
            state.append(row)
        return state

    def a_star_search(state, goal_state, fn):
        queue = []
        entrance = 0
        node = Node(state)
        while not node.is_goal(goal_state):
            node.expand()
            for child in node.children:
                queue_item = (fn(child), entrance, child)
                heapq.heappush(queue, queue_item)
                entrance += 1
            node = heapq.heappop(queue)[2]

        output: List[Any] = [node.state]
        for parent in node.parents():
            output.append(parent.state)
        output.reverse()

        return output

    def id_a_star_search(state, goal_state, fn):

        depth = 0

        def dls(node):
            if node.is_goal(goal_state):
                return node
            if node.depth < depth:
                node.expand()
                for child in node.children:
                    result = dls(child)
                    if result:
                        return result
            return None

        answer = None
        while not answer:
            answer = dls(Node(state))
            depth += 1

        output = [answer.state]
        for parent in answer.parents():
            output.append(parent.state)
        output.reverse()

        return output

    def misplaced_tiles(node):
        misplace_count = 0
        for i in range(len(node.state)):
            for j in range(len(node.state)):
                if node.state[i][j] == 0:
                    continue
                tile_i, tile_j = tiles_places[node.state[i][j]][1]
                if i != tile_i or j != tile_j:
                    misplace_count += 1
        return node.gn() + misplace_count

    def manhattan_variant(node):
        manhattan_distance = 0
        for i in range(len(node.state)):
            for j in range(len(node.state)):
                tile_i, tile_j = tiles_places[node.state[i][j]][1]
                if i != tile_i or j != tile_j:
                    distanceHorizontal = min(abs(tile_i - i), \
                        len(node.state) - abs(tile_i - i))
                    distanceVertical = min(abs(tile_j - j), \
                        len(node.state) - abs(tile_j - j))
                    manhattan_distance += distanceHorizontal + distanceVertical
        return node.gn() + manhattan_distance
