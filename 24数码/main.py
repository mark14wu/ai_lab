from search import Search


# print(len(Search.id_a_star_search(Search.get_state(), Search.goal_state, Search.misplaced_tiles)) - 1)
print(Search.id_a_star_search(Search.get_state(), Search.goal_state, Search.manhattan_variant))
