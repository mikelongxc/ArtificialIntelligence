def _ucbOLD(self) -> StateNode:
    # self._set_t_values()
    # self.frontier.remove(best_ucb_node)
    # self.decrement_frontier_length()

    found_parent = False
    # from root children list, choose best ucb
    best_ucb_node = self._find_max_ucb(self.root_children)
    state = best_ucb_node.state
    c = best_ucb_node.c

    already_explored: List[StateNode] = []

    # if this node has already been traveled through, check explored set
    while best_ucb_node.n > 0:

        # get available random moves from root's child
        moves = list(best_ucb_node.state.moves)
        for i in range(len(self.frontier)):
            if self.frontier[i].parent == best_ucb_node:
                idx_to_rm = self.frontier[i].index
                already_explored.append(self.frontier[i])
                moves.remove(idx_to_rm)
                found_parent = True
        if not found_parent:
            break

        # compare existing ucb to c value. choose best
        best_already_explored_ucb = self._find_max_ucb(already_explored)

        if best_already_explored_ucb.get_ucb() > c:
            best_ucb_node = best_already_explored_ucb
        else:
            rdm_moves_idx = random.randint(0, len(moves))
            best_ucb_node = StateNode(
                moves[rdm_moves_idx], state, best_ucb_node, False)

        found_parent = False

    return best_ucb_node