import heapq
from typing import List, Any

from node import Node


class Search:
    count = 0
    # goal_state = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], \
    #               [16, 17, 18, 19, 20], [21, 22, 23, 24, 0]]
    tiles_places = []

    def config(goal_state):
        Search.goal_state = goal_state
        for i in range(len(goal_state)):
            for j in range(len(goal_state)):
                Search.tiles_places.append((goal_state[i][j], (i, j)))
                # heapq.heappush(tiles_places, (goal_state[i][j], (i, j)))
        Search.tiles_places = sorted(Search.tiles_places, key=lambda x: x[0])

    def a_star_search(state, goal_state, fn):
        queue = []
        entrance = 0
        node = Node(state)
        while not node.is_goal(goal_state):
            Search.count += 1
            # if Search.count % 10000 == 0:
            #     print(Search.count)
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
            if Search.count % 1 == 0:
                print(Search.count)
                print("gn:%d" % node.gn())
                print("fn:%d" % node.fn)
                # print(Node.visited)
            if node.is_goal(goal_state):
                return node
            # print("-----")
            # print("fn:%d" % fn(node))
            if fn(node) < depth:
                node.expand()
                for child in node.children:
                    child.fn = fn(child)
                node.children = sorted(node.children, key=lambda sort_child: sort_child.fn)
                for child in node.children:
                    result = dls(child)
                    if result:
                        return result
            return None

        answer = None
        while not answer:
            answer = dls(Node(state))
            depth += 1
            Node.visited = {}
            # print(depth)

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
        end = len(node.state) - 1

        for i in range(len(node.state)):
            for j in range(len(node.state)):
                if node.state[i][j] == 0:
                    continue
                tile_i, tile_j = Search.tiles_places[node.state[i][j]][1]
                # print("------")
                # print("%d, %d" % (tile_i, tile_j))
                # print("%d, %d" % (i, j))
                if i != tile_i or j != tile_j:
                    naive_distance = abs(tile_i - i) + abs(tile_j - j)

                    left_distance = abs(tile_i - 0) + abs(tile_j - mid) + \
                                    abs(i - end) + abs(j - mid) + 1

                    right_distance = abs(tile_i - end) + abs(tile_j - mid) + \
                                     abs(i - 0) + abs(j - mid) + 1

                    up_distance = abs(tile_i - mid) + abs(tile_j - 0) + \
                                  abs(i - mid) + abs(j - end) + 1

                    down_distance = abs(tile_i - mid) + abs(tile_j - end) + \
                                    abs(i - mid) + abs(j - 0) + 1

                    distances = [naive_distance, left_distance, right_distance, \
                                 up_distance, down_distance]

                    manhattan_distance += min(distances)

        # print(manhattan_distance)
        # print(node.gn())
        # exit(0)
        return node.gn() + 1 * manhattan_distance
