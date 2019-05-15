from search import Search
import time


def get_state(lines):
    state = []
    # n = int(input())
    # n = 5
    for line in lines:
        row = [int(x) for x in line.split('\t')]
        state.append(row)
    return state

def get_route(states):
    init_state = states[0]
    for j in range(len(init_state)):
        for i in range(len(init_state)):
            if init_state[j][i] == 0:
                current_i, current_j = i, j

    route = ""

    mid = int(len(init_state) / 2)
    end = len(init_state) - 1

    for state in states[1:]:
        if current_j - 1 >= 0 and state[current_j - 1][current_i] == 0:
            route += 'U'
            current_j -= 1
        elif current_j == 0 and current_i == mid and state[end][current_i] == 0:
            route += 'U'
            current_j = end
        elif current_j + 1 <= end and state[current_j + 1][current_i] == 0:
            route += 'D'
            current_j += 1
        elif current_j == end and current_i == mid and state[0][current_i] == 0:
            route += 'D'
            current_j = 0
        elif current_i - 1 >= 0 and state[current_j][current_i - 1] == 0:
            route += 'L'
            current_i -= 1
        elif current_i == 0 and current_j == mid and state[current_j][end] == 0:
            route += 'L'
            current_i = end
        elif current_i + 1 <= end and state[current_j][current_i + 1] == 0:
            route += 'R'
            current_i += 1
        elif current_i == end and current_j == mid and state[current_j][0] == 0:
            route += 'R'
            current_i = 0

    return route

def output_to_file(search_fn, heauristics, filename):
    t1 = time.time()
    result_states = search_fn(heauristics)
    t2 = time.time()
    elapsed_time = t2 - t1
    out_file = open(filename, 'w')
    out_file.write('\n'.join([str(elapsed_time) + 's', get_route(result_states), \
                             str(len(result_states) - 1)]))
    out_file.close()

def main():
    Search.config(get_state(open('input.txt').readlines()), \
                  get_state(open('target.txt').readlines()))

    output_to_file(Search.a_star_search, Search.misplaced_tiles, "Ah1_solution.txt")
    output_to_file(Search.id_a_star_search, Search.misplaced_tiles, "IDAh1_solution.txt")
    output_to_file(Search.a_star_search, Search.manhattan_variant, "Ah2_solution.txt")
    output_to_file(Search.id_a_star_search, Search.manhattan_variant, "IDAh2_solution.txt")

    # print(len(Search.a_star_search(Search.manhattan_variant)) - 1)
    # print(Search.count)
    # print(len(Search.id_a_star_search(Search.misplaced_tiles)) - 1)
    # print(len(Search.id_a_star_search(Search.manhattan_variant)) - 1)
    # print(Search.count)

if __name__ == '__main__':
    main()