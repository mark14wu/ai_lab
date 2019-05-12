from search import Search


def get_state(lines):
    state = []
    # n = int(input())
    # n = 5
    for line in lines:
        row = [int(x) for x in line.split('\t')]
        state.append(row)
    return state

def main():
    input_state = get_state(open('input.txt').readlines())
    Search.config(get_state(open('target.txt').readlines()))
    # a_star_h1 = Search.a_star_search(input_state, Search.goal_state, Search.misplaced_tiles)
    print(len(Search.a_star_search(input_state, Search.goal_state, Search.manhattan_variant)) - 1)
    # print(len(Search.id_a_star_search(input_state, Search.goal_state, Search.manhattan_variant)) - 1)
    # print(len(Search.id_a_star_search(input_state, Search.goal_state, Search.misplaced_tiles)) - 1)

if __name__ == '__main__':
    main()