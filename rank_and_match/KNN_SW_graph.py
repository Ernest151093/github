from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict

import numpy as np
from tqdm.auto import tqdm


def distance(pointA: np.ndarray, all_documents: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(pointA - all_documents, axis=1, keepdims=True)
    return dist


def create_sw_graph(
        data: np.ndarray,
        num_candidates_for_choice_long: int = 10,
        num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10,
        num_edges_short: int = 5,
        use_sampling: bool = False,
        sampling_share: float = 0.05,
        dist_f: Callable = distance
    ) -> Dict:
    edges = defaultdict(list)
    num_points = data.shape[0]

    for cur_point_idx in tqdm(range(num_points)):
        if not use_sampling:
            all_dists = dist_f(data[cur_point_idx, :], data)
            argsorted = np.argsort(all_dists.reshape(1, -1))[0][1:]
        else:
            sample_size = int(num_points * sampling_share)
            choiced = np.random.choice(
                list(range(num_points)), size=sample_size, replace=False)
            part_dists = dist_f(data[cur_point_idx, :], data[choiced, :])
            argsorted = choiced[np.argsort(part_dists.reshape(1, -1))[0][1:]]

        short_cands = argsorted[:num_candidates_for_choice_short]
        short_choice = np.random.choice(
            short_cands, size=num_edges_short, replace=False)

        long_cands = argsorted[-num_candidates_for_choice_long:]
        long_choice = np.random.choice(
            long_cands, size=num_edges_long, replace=False)

        for i in np.concatenate([short_choice, long_choice]):
            edges[cur_point_idx].append(i)

    return dict(edges)


def calc_d_and_upd(all_visited_points: OrderedDict, query_point: np.ndarray,
                   all_documents: np.ndarray, point_idx: int, dist_f: Callable
                   ) -> Tuple[float, bool]:
    if point_idx in all_visited_points:
        return all_visited_points[point_idx], True
    cur_dist = dist_f(
        query_point, all_documents[point_idx, :].reshape(1, -1))[0][0]
    all_visited_points[point_idx] = cur_dist
    return cur_dist, False


def nsw(query_point: np.ndarray, all_documents: np.ndarray, graph_edges: Dict,
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    all_visited_points = OrderedDict()
    num_started_points = 0
    # pbar = tqdm(total=num_start_points)
    while ((num_started_points < num_start_points) or (len(all_visited_points) < search_k)):
        # pbar.update(1)
        cur_point_idx = np.random.randint(0, all_documents.shape[0]-1)
        cur_dist, verdict = calc_d_and_upd(
            all_visited_points, query_point, all_documents, cur_point_idx, dist_f)
        if verdict:
            continue

        while True:
            min_dist = cur_dist
            choiced_cand = cur_point_idx

            cands_idxs = graph_edges[cur_point_idx]
            true_verdict_cands = set([cur_point_idx])
            for cand_idx in cands_idxs:
                tmp_d, verdict = calc_d_and_upd(
                    all_visited_points, query_point, all_documents, cand_idx, dist_f)
                if tmp_d < min_dist:
                    min_dist = tmp_d
                    choiced_cand = cand_idx
                if verdict:
                    true_verdict_cands.add(cand_idx)
            else:
                if choiced_cand in true_verdict_cands:
                    break
                cur_dist = min_dist
                cur_point_idx = choiced_cand
                continue
            break
        num_started_points += 1

    best_idxs = np.argsort(list(all_visited_points.values()))[:search_k]
    final_idx = np.array(list(all_visited_points.keys()))[best_idxs]
    return final_idx

