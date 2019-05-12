import heapq
from typing import List, Any

from node import Node


class Search:
    count = 0
    goal_state = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],\
                  [16, 17, 18, 19, 20], [21, 22, 23, 24, 0]]
    tiles_places = []
    for i in range(len(goal_state)):
        for j in range(len(goal_state)):
            tiles_places.append((goal_state[i][j], (i, j)))
            # heapq.heappush(tiles_places, (goal_state[i][j], (i, j)))
    tiles_places = sorted(tiles_places, key=lambda x: x[0])
    # print(tiles_places)

    def get_state():
        state: List[List[int]] = []
        # n = int(input())
        n = 5
        for i in range(n):
            row = [int(x) for x in input().split('\t')]
            state.append(row)
        return state

    def a_star_search(state, goal_state, fn):
        queue = []
        entrance = 0
        node = Node(state)
        while not node.is_goal(goal_state):
            Search.count += 1
            if Search.count % 10000 == 0:
                print(Search.count)
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
            Search.count += 1
            if Search.count % 10000 == 0:
                print(Search.count)
            if node.is_goal(goal_state):
                return node
            # if node.depth + fn(node) < depth:
            if node.depth + fn(node) < depth:
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

        mid = int(len(node.state) / 2)
        # print("mid:%d" % mid)
        end = len(node.state) - 1
        # print("end:%d" % end)

        # print(node.state)

        for i in range(len(node.state)):
            for j in range(len(node.state)):
                tile_i, tile_j = Search.tiles_places[node.state[i][j]][1]
                # print("--------")
                # print("(%d, %d)" % (tile_i, tile_j))
                # print("(%d, %d)" % (i, j))
                if i != tile_i or j != tile_j:
                    naive_distance = abs(tile_i - i) + abs(tile_j - j)

                    left_distance = abs(tile_i - 0) + abs(tile_j - mid) + \
                        abs(i - end) + abs(j - mid) + 1

                    right_distance = abs(tile_i - end) + abs(tile_j - mid) + \
                        abs(i - 0) + abs(j - mid) + 1

                    up_distance = abs(tile_i - mid) + abs(tile_j - 0) + \
                        abs(i - mid) + abs(j - end)

                    down_distance = abs(tile_i - mid) + abs(tile_j - end) + \
                        abs(i - mid) + abs(j - 0)

                    distances = [naive_distance, left_distance, right_distance,\
                                 up_distance, down_distance]

                    # print(distances)

                    manhattan_distance += min(distances)

        # print("md:%d" % manhattan_distance)
        # exit(0)

        return node.gn() + manhattan_distance
