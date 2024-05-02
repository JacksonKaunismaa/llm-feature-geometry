import numpy as np
import itertools


def permute_cost(perm, input_diffs, target_diffs):
    """Compute the Frobenius norm difference of the diffs matrices after permuting the rows and columns of the input diffs matrix
    perm: permutation of the indices (n integers)
    input_diffs: n x n
    target_diffs: n x n"""
    perm_diffs = input_diffs[perm,:][:,perm]
    return ((perm_diffs - target_diffs)**2).sum()


def get_swap_perm(perm, swap):
    """Get the permutation that would result from swapping two indices"""
    perm = list(perm)
    perm[swap[0]], perm[swap[1]] = perm[swap[1]], perm[swap[0]]
    return perm


def swap_perm_cost(swap, input_diffs, target_diffs):
    """Compute the Frobenius norm difference after swapping two indices
    swap: tuple of two indices
    input_diffs: n x n
    target_diffs: n x n"""
    perm = list(range(input_diffs.shape[0]))
    return permute_cost(get_swap_perm(perm, swap), input_diffs, target_diffs)


def greedy_permute(input_diffs, target_diffs, max_iter=1000):
    """Greedy algorithm to find the permutation that minimizes the Frobenius norm difference between the input and target diffs matrices
    At each iteration, it tries all possible swaps and picks the one that minimizes the cost. Each iteration
    runs through the possible swaps in a different order. It returns if the cost is 0 or if no improvement was made.
    input_diffs: n x n
    target_diffs: n x n
    max_iter: maximum number of iterations to run the algorithm
    Returns: permutation of the indices (n integers), cost of the permutation"""
    n = input_diffs.shape[0]
    identity_perm = list(range(n))
    curr_perm = identity_perm
    all_swaps = list(itertools.combinations(range(n), 2))
    for it in range(max_iter):
        np.random.shuffle(all_swaps)
        curr_cost = permute_cost(identity_perm, input_diffs, target_diffs)
        best_swap = None
        best_cost = np.inf
        for (i, j) in all_swaps:#zip(range(n), range(1,n)):
            swap = (i, j)
            perm_cost = swap_perm_cost(swap, input_diffs, target_diffs)
            if perm_cost < best_cost:
                best_swap = swap
                best_cost = perm_cost
            if np.isclose(best_cost, 0):
                break
        
        # swap the two indices
        swap_perm = get_swap_perm(identity_perm, best_swap)
        input_diffs = input_diffs[swap_perm,:][:,swap_perm]
        curr_perm = get_swap_perm(curr_perm, best_swap)
        # print(f"Iteration {it+1}, curr_cost: {curr_cost}, {best_cost=}")

        if np.isclose(best_cost-curr_cost, 0) or np.isclose(best_cost, 0):
            # print(best_swap)
            break
    
    return curr_perm, curr_cost


def poly_order_space(diffs, max_order=5):
    """A particular example of a map from a row of a diffs matrix to a single number. This particular example
    takes the sum of the p-norms of the rows, for p from 1 to max_order.
    diffs: n x n
    max_order: int
    Returns: n dim vector"""
    # given an (n, f) diffs matrix map it to an (n) dim vector that computes some permutation invariant function of the rows
    # it should have the property that small deviations in the input lead to small deviations in output (smoothness)
    # it should have the property that 2 sets of input values/rows being close lead close outputs (continuity)
    totals = np.zeros(diffs.shape[0])
    for order in range(1, max_order+1):
        totals += np.linalg.norm(diffs, ord=order, axis=1)
    return totals
        

def sorting_assignment(input_diffs, target_diffs, mapper):
    """Map the rows of both input diffs matrices into a shared order space using the given `mapper` function.
    The map should should be permutation invariant to the order of the entries in the row. It should vectors that have similar
    sets of entries to similar values. It should map vectors that have different sets of entries to very different values.
    It should probably alse be well-behaved, not blowing up due to large values in the input. Once the rows are mapped into the
    order space, we sort both the input and target rows based on the values in the order space. We can then find the permutation 
    that goes from the input to the target by inverting the permutation of the target and composing it with the permutation of 
    the input.
    input_diffs: n x n
    target_diffs: n x n
    mapper: function that maps an n x n diffs matrix to an n dim vector"""
    input_order = mapper(input_diffs)
    target_order = mapper(target_diffs)

    def perm_inv(p):  # invert the permutation
        p = list(p)
        return [p.index(i) for i in range(len(p))]

    target_permutation = np.argsort(input_order)
    input_permutation = np.argsort(target_order)
    return input_permutation[perm_inv(target_permutation)]


def combined_sorting(input_diffs, target_diffs, mapper, max_greedy_iter=1000):
    """Combine the map-sort based and greedy algorithms to try to find the permutation that minimizes the Frobenius norm difference.
    input_diffs: n x n
    target_diffs: n x n
    mapper: function that maps an n x n diffs matrix to an n dim vector"""
    sort_perm = sorting_assignment(input_diffs, target_diffs, mapper)
    sort_cost = permute_cost(sort_perm, input_diffs, target_diffs)
    # print("Sorting done, cost is ", sort_cost)

    if np.isclose(sort_cost, 0):
        # print("Skipping greedy, since sorting is already perfect")
        return sort_perm
    
    perm_diffs = input_diffs[sort_perm,:][:,sort_perm]
    greedy_perm,_ = greedy_permute(perm_diffs, target_diffs, max_iter=max_greedy_iter)
    # print("Greedy done, cost is ", permute_cost(greedy_perm, perm_diffs, target_diffs))
    # compose the 2 permutations to get the final permutation
    return sort_perm[greedy_perm]