import numpy as np
import copy


class CrossOver:
    def __init__(self):
        pass

    def is_ready_to_reproduce(self, agent_ind, neighbourhood_inds, grid):
        ready_to_reproduce = False
        neighbourhood_resources = [grid[i].resources for i in neighbourhood_inds if i != agent_ind]

        if grid[agent_ind].resources > max(neighbourhood_resources):
            ready_to_reproduce = True

        return ready_to_reproduce

    def run(self, grid, new_grid_cands):

        new_grid = []
        for i in range(len(grid)):
            if len(new_grid_cands[i]) == 0:
                new_grid.append(copy.deepcopy(grid[i]))
            elif len(new_grid_cands[i]) == 1:
                new_grid.append(copy.deepcopy(grid[new_grid_cands[i][0]]))
            else:
                candidates = [grid[cand_i].energy for cand_i in new_grid_cands[i]]
                best_candidate = np.argmax(candidates)
                new_grid.append(copy.deepcopy(grid[new_grid_cands[i][best_candidate]]))

        return new_grid
