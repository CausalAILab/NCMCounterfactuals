from src.ds import CausalGraph, identify, graph_search

import random


def sample_cg(n, dir_rate, bidir_rate, enforce_direct_path=False, enforce_bidirect_path=False, enforce_ID=None):
    """
    Samples a random causal diagram with n variables, including X and Y.
    All directed edges are independently included with a chance of dir_rate.
    All bidirected edges are independently included with a chance of bidir_rate.
    enforce_direct_path: if True, then there is guaranteed to be a directed path from X to Y
        this implies almost surely that P(Y | do(X)) != P(Y)
    enforce_bidirect_path: if True, then there is guaranteed to be a bidirected path from X to Y
        this implies P(Y | do(X)) is not amenable to backdoor adjustment
    enforce_ID: if True, then P(Y | do(X)) is guaranteed to be identifiable
                if False, then P(Y | do(X)) is guaranteed to not be identifiable
    """
    cg = None
    done = False

    while not done:
        x_loc = random.randint(0, n - 2)
        V_list = ['V{}'.format(i + 1) for i in range(n - 2)]
        V_list.insert(x_loc, 'X')
        V_list.append('Y')

        de_list = []
        be_list = []
        for i in range(len(V_list) - 1):
            for j in range(i + 1, len(V_list)):
                if random.random() < dir_rate:
                    de_list.append((V_list[i], V_list[j]))
                if random.random() < bidir_rate:
                    be_list.append((V_list[i], V_list[j]))

        cg = CausalGraph(V_list, de_list, be_list)

        done = True
        if enforce_direct_path and not graph_search(cg, 'X', 'Y', edge_type="direct"):
            done = False
        if enforce_bidirect_path and not graph_search(cg, 'X', 'Y', edge_type="bidirect"):
            done = False

        if enforce_ID is not None:
            id_status = (identify(X={'X'}, Y={'Y'}, G=cg) != "FAIL")
            if enforce_ID != id_status:
                done = False

    return cg


if __name__ == "__main__":
    # cg = CausalGraph.read("../../dat/cg/napkin.cg")
    cg = sample_cg(10, 0.3, 0.2, enforce_direct_path=False, enforce_bidirect_path=False, enforce_ID=False)
    result = identify(X={'X'}, Y={'Y'}, G=cg)
    print(result)
    if result != "FAIL":
        print(result.get_latex())

