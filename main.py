import os
import sys
import numpy as np
import extract


def features_to_string(features):
    energies = [f["energy"] for f in features]
    low_th = np.percentile(energies, 33)
    high_th = np.percentile(energies, 66)
    peak_th = np.percentile(energies, 85)

    symbols = []
    for f in features:
        e = f["energy"]
        if e >= peak_th:
            sym = "D"
        elif e >= high_th:
            sym = "C"
        elif e >= low_th:
            sym = "B"
        else:
            sym = "A"
        symbols.append(sym)
    return symbols


GENRE_NFAS = {
    "Rock": {
        "start": "q0",
        "accept": {"q_chorus", "q_solo", "q_outro"},
        "transitions": {
            ("q0", "A"): ["q_intro"],
            ("q0", "B"): ["q_verse"],
            ("q_intro", "A"): ["q_intro"],
            ("q_intro", "B"): ["q_verse"],
            ("q_verse", "A"): ["q_verse"],
            ("q_verse", "B"): ["q_verse"],
            ("q_verse", "C"): ["q_chorus"],
            ("q_chorus", "C"): ["q_chorus"],
            ("q_chorus", "D"): ["q_chorus", "q_solo"],
            ("q_chorus", "B"): ["q_verse"],
            ("q_solo", "D"): ["q_solo"],
            ("q_solo", "C"): ["q_chorus"],
            ("q_chorus", "A"): ["q_outro"],
            ("q_outro", "A"): ["q_outro"],
        },
    },
    "EDM": {
        "start": "q0",
        "accept": {"q_drop", "q_outro"},
        "transitions": {
            ("q0", "A"): ["q_intro"],
            ("q0", "B"): ["q_build"],
            ("q_intro", "A"): ["q_intro"],
            ("q_intro", "B"): ["q_build"],
            ("q_build", "B"): ["q_build"],
            ("q_build", "C"): ["q_build"],
            ("q_build", "D"): ["q_drop"],
            ("q_drop", "D"): ["q_drop"],
            ("q_drop", "C"): ["q_drop"],
            ("q_drop", "A"): ["q_break"],
            ("q_break", "A"): ["q_break"],
            ("q_break", "B"): ["q_build"],
            ("q_break", "A"): ["q_outro"],
            ("q_outro", "A"): ["q_outro"],
        },
    },
    "Pop": {
        "start": "q0",
        "accept": {"q_chorus", "q_outro"},
        "transitions": {
            ("q0", "A"): ["q_intro"],
            ("q0", "B"): ["q_verse"],
            ("q_intro", "A"): ["q_intro"],
            ("q_intro", "B"): ["q_verse"],
            ("q_verse", "A"): ["q_verse"],
            ("q_verse", "B"): ["q_verse"],
            ("q_verse", "C"): ["q_chorus"],
            ("q_chorus", "C"): ["q_chorus"],
            ("q_chorus", "D"): ["q_chorus"],
            ("q_chorus", "B"): ["q_verse"],
            ("q_chorus", "A"): ["q_outro"],
            ("q_outro", "A"): ["q_outro"],
        },
    },
    "Classical": {
        "start": "q0",
        "accept": {"q_theme", "q_coda"},
        "transitions": {
            ("q0", "A"): ["q_quiet"],
            ("q0", "B"): ["q_theme"],
            ("q_quiet", "A"): ["q_quiet"],
            ("q_quiet", "B"): ["q_theme"],
            ("q_theme", "A"): ["q_theme"],
            ("q_theme", "B"): ["q_theme"],
            ("q_theme", "C"): ["q_crescendo"],
            ("q_crescendo", "C"): ["q_crescendo"],
            ("q_crescendo", "B"): ["q_theme"],
            ("q_crescendo", "A"): ["q_coda"],
            ("q_coda", "A"): ["q_coda"],
        },
    },
}


def simulate_nfa(nfa, string):
    current_states = {nfa["start"]}
    valid_steps = 0
    strict = True

    for ch in string:
        next_states = set()
        for st in current_states:
            next_states.update(nfa["transitions"].get((st, ch), []))

        if next_states:
            current_states = next_states
            valid_steps += 1
        else:
            strict = False
            restarted = set(nfa["transitions"].get((nfa["start"], ch), []))
            current_states = restarted if restarted else {nfa["start"]}

    match_rate = (valid_steps / len(string)) * 100
    reached_accept = bool(current_states & nfa["accept"])
    score = match_rate + (10 if reached_accept else 0)
    return round(score, 2), match_rate, strict and reached_accept


def plot_nfa(nfa_name, nfa_dict, output_file="automata.png"):
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        edge_labels = {}

        for (src, sym), dsts in nfa_dict["transitions"].items():
            for dst in dsts:
                G.add_edge(src, dst)
                key = (src, dst)
                edge_labels[key] = f"{edge_labels.get(key, '')},{sym}".strip(",")

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(8, 5))

        colors = []
        for node in G.nodes():
            if node == nfa_dict["start"]:
                colors.append("#85e385")
            elif node in nfa_dict["accept"]:
                colors.append("#ffd166")
            else:
                colors.append("#90e0ef")

        nx.draw(G, pos, with_labels=True, node_color=colors, node_size=2000, font_weight="bold", arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
        plt.title(f"{nfa_name} NFA")
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    feats = extract.all_features

    symbols = features_to_string(feats)
    w = "".join(symbols)
    print(f"\nInput String w in Sigma*:\n{w}\n")

    print(f"{'Genre':<12} | {'Score':<8} | {'Match %':<8} | {'Strict L(M)'}")
    print("-" * 42)

    results = {}
    for genre, nfa in GENRE_NFAS.items():
        score, match_pct, is_strict = simulate_nfa(nfa, symbols)
        results[genre] = score
        strict_str = "ACCEPTED" if is_strict else "REJECTED"
        print(f"{genre:<12} | {score:<8.1f} | {match_pct:<7.1f}% | {strict_str}")

    winner = max(results, key=results.get)
    print("\n" + "=" * 40)
    print(f"PREDICTED GENRE: {winner.upper()}")
    print("=" * 40 + "\n")

    plot_nfa(winner, GENRE_NFAS[winner], f"{winner.lower()}_nfa.png")
