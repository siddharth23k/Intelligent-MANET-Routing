import os
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import joblib
import pandas as pd

from .data_utils import DATA_PATH, RESULTS_DIR

# Configuration
SOURCE = 10
DESTINATION = 0
POP_SIZE = 20
NUM_GENERATIONS = 50
MAX_ROUTE_LENGTH = 6
MUTATION_RATE = 0.1

# Load dataset and trained model
data = pd.read_csv(DATA_PATH)

model_path = os.path.join(RESULTS_DIR, "random_forest_model.pkl")
model = joblib.load(model_path)


def _build_neighbor_map(df: pd.DataFrame) -> Dict[int, List[int]]:
    """Build an undirected neighbor map from node to list of neighbors."""
    neighbor_map: Dict[int, set] = defaultdict(set)
    for _, row in df[["nodeA", "nodeB"]].drop_duplicates().iterrows():
        a = int(row["nodeA"])
        b = int(row["nodeB"])
        neighbor_map[a].add(b)
        neighbor_map[b].add(a)
    return {node: sorted(neighbors) for node, neighbors in neighbor_map.items()}


def _build_edge_features(df: pd.DataFrame) -> Dict[Tuple[int, int], List[float]]:
    """
    For each node pair (nodeA, nodeB) keep the latest observation
    of [distance, time] to use as input to the ML model.
    """
    features: Dict[Tuple[int, int], List[float]] = {}
    grouped = df.sort_values("time").groupby(["nodeA", "nodeB"])

    for (a, b), group in grouped:
        latest = group.iloc[-1]
        key = (int(a), int(b))
        features[key] = [
            float(latest["distance"]),
            float(latest["time"]),
        ]

    return features


NEIGHBOR_MAP = _build_neighbor_map(data)
EDGE_FEATURES = _build_edge_features(data)


def generate_route() -> List[int]:
    """
    Generate a feasible route from SOURCE to DESTINATION by
    walking along known neighbors, avoiding revisiting nodes.
    """
    route = [SOURCE]

    while route[-1] != DESTINATION and len(route) < MAX_ROUTE_LENGTH:
        current = route[-1]
        neighbors = [n for n in NEIGHBOR_MAP.get(current, []) if n not in route]

        if not neighbors:
            # No unseen neighbors left; force destination and stop.
            break

        next_node = random.choice(neighbors)
        route.append(next_node)

    if route[-1] != DESTINATION:
        route.append(DESTINATION)

    return route


def route_fitness(route: Sequence[int]) -> float:
    """
    Score a route by summing predicted link stability along its hops.
    Routes with more stable links and fewer missing edges score higher.
    """
    score = 0.0

    for i in range(len(route) - 1):
        node_a = route[i]
        node_b = route[i + 1]

        key = (min(node_a, node_b), max(node_a, node_b))
        sample = EDGE_FEATURES.get(key)

        if sample is None:
            # No data for this hop; treat as unstable.
            continue

        pred = model.predict([sample])[0]
        score += float(pred)

    # Simple penalty for very long routes.
    if len(route) > 2:
        score -= 0.1 * (len(route) - 2)

    return score


def evaluate_population(population: Sequence[Sequence[int]]) -> List[List[int]]:
    """Return a new list of routes sorted by fitness (best first)."""
    return sorted(population, key=route_fitness, reverse=True)


def select_parents(population: Sequence[Sequence[int]], num_parents: int) -> List[List[int]]:
    """Select the top-performing routes as parents."""
    return list(population[:num_parents])


def crossover(parent1: Sequence[int], parent2: Sequence[int]) -> List[int]:
    """Single-point crossover that preserves SOURCE and DESTINATION."""
    cut1 = max(1, len(parent1) // 2)
    cut2 = max(1, len(parent2) // 2)

    middle = []
    for node in parent2[cut2:-1]:
        if node not in (SOURCE, DESTINATION) and node not in middle:
            middle.append(node)

    child = [SOURCE] + parent1[1:cut1] + middle + [DESTINATION]
    # Enforce maximum length (including endpoints)
    if len(child) > MAX_ROUTE_LENGTH + 1:
        child = child[: MAX_ROUTE_LENGTH] + [DESTINATION]
    return child


def mutate(route: List[int]) -> List[int]:
    """Randomly tweak intermediate hops while respecting neighbor structure."""
    if len(route) <= 2:
        return route

    for i in range(1, len(route) - 1):
        if random.random() < MUTATION_RATE:
            current = route[i - 1]
            candidates = [n for n in NEIGHBOR_MAP.get(current, []) if n not in route]
            if candidates:
                route[i] = random.choice(candidates)

    return route


def run_ga() -> Tuple[List[int], float, List[float]]:
    """Run the genetic algorithm and return best route, its fitness, and history."""
    population = [generate_route() for _ in range(POP_SIZE)]
    best_history: List[float] = []

    for generation in range(NUM_GENERATIONS):
        population = evaluate_population(population)
        best_score = route_fitness(population[0])
        best_history.append(best_score)

        if generation % 10 == 0 or generation == NUM_GENERATIONS - 1:
            print(f"Generation {generation}: best fitness = {best_score:.3f}")

        parents = select_parents(population, POP_SIZE // 2)

        children: List[List[int]] = []
        while len(children) < POP_SIZE - len(parents):
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            children.append(child)

        population = parents + children

    population = evaluate_population(population)
    best_route = population[0]
    best_score = route_fitness(best_route)
    return best_route, best_score, best_history


if __name__ == "__main__":
    best_route, best_score, history = run_ga()
    print("\nBest Route Found:")
    print(best_route)
    print("Route Fitness:", best_score)
