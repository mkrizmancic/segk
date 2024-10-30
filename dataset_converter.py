import pathlib

import torch_geometric.utils as pygUtils
from torch_geometric.loader import DataLoader

from gnn_fiedler_approx.algebraic_connectivity_dataset import ConnectivityDataset
from gnn_fiedler_approx.gnn_utils.utils import create_graph_vis
from my_graphs_dataset import GraphDataset


def inspect_dataset(dataset, num_graphs=1):
    for data in dataset:
        print()
        print(data)
        print("=============================================================")

        # Gather some statistics about the first graph.
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges / 2}")
        print(f"Algrebraic connectivity: {data.y.item():.5f}")
        print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")
        print(f"Features: {data.x}")


def convert_dataset():
    selected_graph_sizes = {
        3: -1,
        4: -1,
        5: -1,
        6: 200,
        7: 200,
        8: 200,
    }

    # Load the dataset.
    root = pathlib.Path("/home/marko/PROJECTS/Topocon_GNN/gnn_fiedler_approx/Dataset")
    graphs_loader = GraphDataset(selection=selected_graph_sizes)
    dataset = ConnectivityDataset(root, graphs_loader)


    # Batch and load data.
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)  # type: ignore

    # If the whole dataset fits in memory, we can use the following lines to get a single large batch.
    batch = next(iter(loader))

    # Create output directory if it doesn't exist
    output_dir = pathlib.Path("/home/marko/PROJECTS/segk/datasets/graph_regression/GRAPHS")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write edgelist to GRAPHS_A.txt
    with open(output_dir / "GRAPHS_A.txt", "w") as f:
        for edge_index in batch.edge_index.t().tolist():
            f.write(f"{edge_index[0]+1}, {edge_index[1]+1}\n")

    # Write graph indicator to GRAPHS_graph_indicator.txt
    with open(output_dir / "GRAPHS_graph_indicator.txt", "w") as f:
        for i, graph_indicator in enumerate(batch.batch.tolist()):
            f.write(f"{graph_indicator}\n")

    # Write graph labels to GRAPHS_graph_labels.txt
    with open(output_dir / "GRAPHS_graph_labels.txt", "w") as f:
        for label in batch.y.tolist():
            f.write(f"{label}\n")

    return dataset, batch


if __name__ == "__main__":
    dataset, batch = convert_dataset()

    # inspect_dataset(dataset)
    # G = pygUtils.to_networkx(batch)
    # fig = create_graph_vis(G)
    # fig.show()
