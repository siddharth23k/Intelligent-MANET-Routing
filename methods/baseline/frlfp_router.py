import numpy as np
import networkx as nx


class FRLFPRouter:

    def build_graphs(self, snapshot, radius=150.0):
        rows = snapshot.to_dict("records")
        node_lfp = {int(r["node_id"]): float(r["lfp"]) for r in rows}
        node_thr = {int(r["node_id"]): float(r["lfp_threshold"]) for r in rows}
        risky = {nid for nid in node_lfp if node_lfp[nid] > node_thr[nid]}

        G = nx.Graph()
        G_filtered = nx.Graph()
        for r in rows:
            nid = int(r["node_id"])
            G.add_node(nid)
            if nid not in risky:
                G_filtered.add_node(nid)

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                u, v = int(rows[i]["node_id"]), int(rows[j]["node_id"])
                dist = np.sqrt((rows[i]["x"] - rows[j]["x"]) ** 2 + (rows[i]["y"] - rows[j]["y"]) ** 2)
                if dist <= radius:
                    rel = float(np.clip(1.0 - max(node_lfp[u], node_lfp[v]), 0.001, 0.999))
                    G.add_edge(u, v, weight=1.0, reliability=rel)
                    if (u not in risky) and (v not in risky):
                        G_filtered.add_edge(u, v, weight=1.0, reliability=rel)

        pos = {int(r["node_id"]): (float(r["x"]), float(r["y"])) for r in rows}
        return G, G_filtered, pos, risky

    @staticmethod
    def route(G_full, G_filtered, src, dst):
        # Prefer filtered graph, fallback to full graph.
        try:
            return nx.shortest_path(G_filtered, src, dst)
        except Exception:
            try:
                return nx.shortest_path(G_full, src, dst)
            except Exception:
                return None

    @staticmethod
    def route_metrics(G, path):
        if not path or len(path) < 2:
            return {"avg_reliability": 0.0, "min_reliability": 0.0, "hop_count": 0}
        rels = [G[u][v]["reliability"] for u, v in zip(path[:-1], path[1:])]
        return {
            "avg_reliability": float(np.mean(rels)),
            "min_reliability": float(np.min(rels)),
            "hop_count": int(len(path) - 1),
        }
