"""This module implements the SimplicialPathsLifting class."""

import networkx as nx
import numpy as np
import pyflagsercount as pfc
import torch
import torch_geometric
from toponetx.classes import CombinatorialComplex

from topobenchmark.transforms.liftings.graph2combinatorial.base import (
    Graph2CombinatorialLifting,
)


class SimplicialPathsLifting(Graph2CombinatorialLifting):
    """Lift graphs to combinatorial complex domain by identifying simplicial paths as simplices.

    This method constructs a combinatorial complex by identifying paths in a graph that correspond to simplices of varying dimensions. The lifting process enables the representation of higher-order relationships in the graph structure.

    Parameters
    ----------
    d1 : int
        The minimum dimension of simplicial paths to consider.
    d2 : int
        The maximum dimension of simplicial paths to consider.
    q : float
        A parameter controlling path selection criteria, such as quality or weight.
    i : int
        Starting node index for path identification.
    j : int
        Ending node index for path identification.
    complex_dim : int, optional
        The maximum dimension of the combinatorial complex. Default is 2.
    chunk_size : int, optional
        The size of chunks for processing large graphs. Default is 1024.
    threshold : float, optional
        A threshold value to filter paths based on specific criteria. Default is 1.
    **kwargs : dict, optional
        Additional keyword arguments for customization.

    Notes
    -----
    This lifting approach is useful for analyzing higher-order structures in networks by extending beyond pairwise relationships to simplicial complexes. It can be applied to study topological prop
    """

    def __init__(
        self,
        d1,
        d2,
        q,
        i,
        j,
        complex_dim=2,
        chunk_size=1024,
        threshold=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d1 = d1
        self.d2 = d2
        self.q = q
        self.i = i
        self.j = j
        self.complex_dim = complex_dim
        self.chunk_size = chunk_size
        self.threshold = threshold

    def _get_complex_connectivity(
        self, combinatorial_complex, adjacencies, incidences, max_rank
    ):
        """Return the connectivity information of the combinatorial complex.

        Parameters
        ----------
        combinatorial_complex : CombinatorialComplex
            The combinatorial complex.
        adjacencies : List[List[int]]
            The list of adjacency pairs.
        incidences : List[List[int]]
            The list of incidence pairs.
        max_rank : int
            The maximum rank of the complex.

        Returns
        -------
        dict
            The connectivity information of the combinatorial complex.
        """
        practical_shape = list(
            np.pad(
                list(combinatorial_complex.shape),
                (0, max_rank + 1 - len(combinatorial_complex.shape)),
            )
        )
        connectivity = {}
        connectivity["shape"] = practical_shape
        for adj in adjacencies:
            connectivity_info = "adjacency"
            if adj[0] < adj[1]:
                connectivity[f"{connectivity_info}_{adj[0]}_{adj[1]}"] = (
                    torch.from_numpy(
                        combinatorial_complex.adjacency_matrix(
                            adj[0], adj[1]
                        ).todense()
                    )
                    .to_sparse()
                    .float()
                )
            else:
                connectivity[f"{connectivity_info}_{adj[0]}_{adj[1]}"] = (
                    torch.from_numpy(
                        combinatorial_complex.coadjacency_matrix(
                            adj[0], adj[1]
                        ).todense()
                    )
                    .to_sparse()
                    .float()
                )
        for inc in incidences:
            connectivity_info = "incidence"
            connectivity[f"{connectivity_info}_{inc[0]}_{inc[1]}"] = (
                torch.from_numpy(
                    combinatorial_complex.incidence_matrix(
                        inc[0], inc[1]
                    ).todense()
                )
                .to_sparse()
                .float()
            )
        return connectivity

    def _get_lifted_topology(
        self, combinatorial_complex: CombinatorialComplex, graph: nx.Graph
    ) -> dict:
        """Return the lifted topology.

        Parameters
        ----------
        combinatorial_complex : CellComplex
            The combinatorial complex representing the lifted structure.
        graph : nx.Graph
            The input graph.

        Returns
        -------
        dict
            The lifted topology.

        Notes
        -----
        This method computes and returns the lifted topology based on the input
        combinatorial complex and the original graph structure.
        """

        adjacencies = [[0, 1]]
        incidences = [[0, 2]]
        lifted_topology = self._get_complex_connectivity(
            combinatorial_complex, adjacencies, incidences, self.complex_dim
        )

        feat = torch.stack(
            list(nx.get_node_attributes(graph, "features").values())
        )
        lifted_topology["x_0"] = feat
        lifted_topology["x_2"] = torch.matmul(
            lifted_topology["incidence_0_2"].t(), feat
        )

        return lifted_topology

    def _create_flag_complex_from_dataset(self, dataset, complex_dim=2):
        """Create a directed flag complex from a dataset.

        Parameters
        ----------
        dataset : torch_geometric.data.Data
            The input dataset.
        complex_dim : int, optional
            The maximum dimension of the complex. Default is 2.

        Returns
        -------
        DirectedQConnectivity
            The directed flag complex.
        """
        dataset_digraph = nx.DiGraph()

        dataset_digraph.add_edges_from(
            list(
                zip(
                    dataset.edge_index[0].tolist(),
                    dataset.edge_index[1].tolist(),
                    strict=False,
                )
            )
        )

        return DirectedQConnectivity(dataset_digraph, complex_dim)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        """Lift the graph to a combinatorial complex by identifying simplicial paths.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted.
        """
        FlG = self._create_flag_complex_from_dataset(data, complex_dim=2)

        indices = FlG.qij_adj(
            FlG.complex[self.d1],
            FlG.complex[self.d2],
            self.q,
            self.i,
            self.j,
            self.chunk_size,
        )

        G = self._generate_graph_from_data(data)
        paths = FlG.find_paths(indices, self.threshold)

        cc = CombinatorialComplex(G)

        for p in paths:  # retrieve nodes that compose each path
            cell = list()
            for c in p:
                cell += list(FlG.complex[2][c].numpy())
            cell = list(set(cell))  # remove duplicates

            cc.add_cell(cell, rank=2)

        return self._get_lifted_topology(cc, G)


class DirectedQConnectivity:
    """Compute directed flag complex and q-connectivity of directed graphs.

    Parameters
    ----------
    digraph : nx.DiGraph
        The directed graph to compute the directed flag complex of.
    complex_dim : int
        The maximum dimension of the complex to compute.
    flagser_num_threads : int, optional
        The number of threads to use in the flagser computation. Default is 4.

    Notes
    -----
    Let G=(V,E) be a directed graph. The directed flag complex of G, dFl(G),
    is the ordered simplicial complex whose k-simplices vertices are all
    totally ordered (k+1)-cliques, i.e. (v_0, ..., v_n) such that (v_i, v_j) ∈ E
    for all i ≤ j. This class provides a way to compute the directed flag
    complex of a directed graph, compute the qij-connectivity of the complex
    and find the maximal simplicial paths arising from the qij-connectivity.

    References
    ----------
    .. [1] Henri Riihïmaki. Simplicial q-Connectivity of Directed Graphs
        with Applications to Network Analysis. doi:10.1137/22M1480021.

    .. [2] D. Lütgehetmann, D. Govc, J.P. Smith, and R. Levi. Computing
        persistent homology of directed flag complexes. arXiv:1906.10458.
    """

    complex: dict[int, set[tuple]]

    def __init__(
        self,
        digraph: nx.DiGraph,
        complex_dim: int = 2,
        flagser_num_threads: int = 4,
    ):
        self.digraph = digraph
        self.complex_dim = complex_dim

        sparse_adjacency_matrix = nx.to_scipy_sparse_array(
            digraph, format="csr"
        )

        self.X = pfc.flagser_count(
            sparse_adjacency_matrix,
            threads=flagser_num_threads,
            return_simplices=True,
            max_dim=self.complex_dim,
            compressed=False,
        )

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        # else torch.device("cpu") #Server

        # self.device = torch.device("mps")  # my macbook
        self.complex = self.X["simplices"]

        self.complex[0] = torch.tensor(
            [[node] for node in digraph.nodes], device=self.device
        )
        self.complex[1] = torch.tensor(
            [list(edge) for edge in digraph.edges], device=self.device
        )
        self.complex = [
            torch.tensor(item, device=self.device) if i >= 2 else item
            for i, item in enumerate(self.complex)
        ]

    def _d_i_batched(self, i: int, simplices: torch.tensor) -> torch.tensor:
        """Compute face map d_i of simplices in batched tensor.

        Parameters
        ----------
        i : int
            The index of the face map.
        simplices : torch.Tensor
            The batch of simplices.
            - Shape: (batch_size, n_vertices)

        Returns
        -------
        torch.Tensor
            The batch of simplices after applying the face map d_i.
            - Shape: (batch_size, n_vertices-1)

        Notes
        -----
        The map d_i removes a vertex at position min{i, dim(σ)} for each simplex σ
        in the batch, where dim(σ) is the dimension of the simplex.

        References
        ----------
        .. [1] Henri Riihïmaki. Simplicial q-Connectivity of Directed
            Graphs with Applications to Network Analysis. doi:10.1137/22M1480021.
        """

        batch_size, n_vertices = simplices.shape
        indices = torch.arange(
            n_vertices, device=simplices.device
        )  # Allocated on the same device as `simplices`
        # Create a mask that excludes the i-th vertex
        mask = indices != min(i, n_vertices - 1)
        # Use advanced indexing to select vertices while preserving the
        # batch structure
        return simplices[:, mask]

    def _gen_q_faces_batched(
        self, simplices: torch.tensor, c: int
    ) -> torch.tensor:
        """Compute q-dimensional faces of simplices in batched tensor.

        Parameters
        ----------
        simplices : torch.Tensor
            The simplices tensor. Shape: (batch_size, n_vertices).
        c : int
            The cardinality of the faces to compute.

        Returns
        -------
        torch.Tensor
            The q-dimensional faces of the simplices tensor.
            Shape: (batch_size, n_faces, c).

        Notes
        -----
        This function computes the q-dimensional faces of the simplices in the
        batched simplices tensor, where c represents the cardinality of the faces
        to compute and q = c - 1.
        """

        combinations = torch.combinations(
            torch.tensor(range(simplices.size(1)), device=self.device), c
        )

        return simplices[:, combinations]

    def _multiple_contained_chunked(
        self, sigmas: torch.Tensor, taus: torch.Tensor, chunk_size: int = 1024
    ) -> torch.Tensor:
        """Compute chunked adjacency matrix for simplex containment relation.

        Parameters
        ----------
        sigmas : torch.Tensor
            The first simplices tensor. Shape: (Ns, cs).
        taus : torch.Tensor
            The second simplices tensor. Shape: (Nt, ct).
        chunk_size : int, optional
            The size of the chunks to process. Default is 1024.

        Returns
        -------
        torch.sparse_coo_tensor
            Adjacency matrix A where A(i,j) = 1 if sigma_i ⊆ tau_j, and 0 otherwise.
            Shape: (Ns, Nt).

        Notes
        -----
        This function computes the adjacency matrix induced by the relation
        sigma_i ⊆ tau_j. It uses chunking to avoid memory issues when processing
        large datasets.
        """

        Ns, cs = sigmas.size()
        Nt, ct = taus.size()

        # If cs > ct, no sigma can be contained in any tau.
        if cs > ct:
            return torch.sparse_coo_tensor(
                torch.empty([2, 0], dtype=torch.long),
                [],
                size=(Ns, Nt),
                dtype=torch.bool,
            )

        # Generate faces of taus
        faces = self._gen_q_faces_batched(taus, cs)
        Nf = faces.size(1)
        total_faces = Nt * Nf

        indices = []

        # Process in chunks for memory efficiency purposes.
        for i in range(0, Ns, chunk_size):
            end_i = min(i + chunk_size, Ns)
            sigmas_chunk = sigmas[i:end_i]  # Shape: [min(chunk_size,
            # remaining Ns), cs]

            temp_true_indices = []

            # Compute diffs and matches for this chunk
            for j in range(0, total_faces, chunk_size):
                end_j = min(j + chunk_size, total_faces)
                faces_chunk = faces.view(-1, cs)[
                    j:end_j
                ]  # Shape: [min(chunk_size, remaining faces), cs]

                # Broadcasting happens here with much smaller tensors
                diffs = sigmas_chunk.unsqueeze(1) - faces_chunk.unsqueeze(
                    0
                )  # shape: [min(chunk_size, remaining Ns), min(chunk_size,
                # remaining faces), cs]

                matches = (
                    diffs.abs().sum(dim=2) == 0
                )  # shape: [min(chunk_size, remaining Ns),  min(chunk_size,
                # remaining faces)]

                # (end_i - i) is the number of sigmas in the chunk.
                # (end_j - j) // Nf is the number of taus in the chunk
                # Nf is the number of faces in each tau of dimension equal
                # to the dimension of the simplices in sigma.
                matches_reshaped = matches.view(
                    end_i - i, (end_j - j) // Nf, Nf
                )

                matches_aggregated = matches_reshaped.any(dim=2)

                # Update temporary result for this chunk of sigmas
                if matches_aggregated.nonzero(as_tuple=False).size(0) > 0:
                    temp_indices = matches_aggregated.nonzero(as_tuple=False).T
                    temp_indices[0] += i  # Adjust sigma indices for chunk
                    # offset
                    temp_indices[1] += j // Nf  # Adjust tau indices for
                    # chunk offset
                    temp_true_indices.append(temp_indices)

            if temp_true_indices:
                indices.append(torch.cat(temp_true_indices, dim=1))

        if indices:
            indices = torch.cat(indices, dim=1)
        else:
            indices = torch.empty([2, 0], dtype=torch.long)

        return torch.sparse_coo_tensor(
            indices,
            torch.ones(indices.size(1), dtype=torch.bool),
            size=(Ns, Nt),
            device="cpu",
        )

    def _alpha_q_contained_sparse(
        self,
        sigmas: torch.Tensor,
        taus: torch.Tensor,
        q: int,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        """Compute the adjacency matrix induced by the relation :math:`\sigma_i \sim \tau_j \Leftrightarrow \exists \alpha_q \subseteq \sigma_i \cap \tau_j`. This function is chunked to avoid memory issues.

        Parameters
        ----------
        sigmas : torch.Tensor
            The first simplices tensor with shape (Ns, cs).
        taus : torch.Tensor
            The second simplices tensor with shape (Nt, ct).
        q : int
            The dimension of the alpha_q simplices.
        chunk_size : int, optional
            The size of the chunks to process. Default is 1024.

        Returns
        -------
        torch.sparse_coo_tensor
            Adjacency matrix with shape (Ns, Nt). :math:`A(i,j) = 1` if
            there exists :math:`\alpha_q \in \Sigma_q` such that
            :math:`\alpha_q \subseteq \sigma_i \cap \tau_j:`, and :math:`0`
            otherwise.
        """

        alpha_q_in_sigmas = self._multiple_contained_chunked(
            self.complex[q], sigmas, chunk_size
        ).to(torch.float)

        alpha_q_in_taus = self._multiple_contained_chunked(
            self.complex[q], taus, chunk_size
        ).to(torch.float)

        # Compute the intersection of the two sparse tensors to get the
        # alpha_q contained in both sigmas and taus

        intersect = torch.sparse.mm(alpha_q_in_sigmas.t(), alpha_q_in_taus)

        values = torch.ones(intersect._indices().size(1))

        return torch.sparse_coo_tensor(
            intersect._indices(),
            values,
            dtype=torch.bool,
            size=(sigmas.size(0), taus.size(0)),
        )

    def qij_adj(
        self,
        sigmas: torch.tensor,
        taus: torch.tensor,
        q: int,
        i: int,
        j: int,
        chunk_size: int = 1024,
    ):
        """
        Compute the adjacency matrix associated with the (q, d_i, d_j)-connectivity relation of two not necessarily distinct pairs of skeletons of the complex.

        Parameters
        ----------
        sigmas : torch.Tensor
            The first batch of simplices corresponds to a skeleton of the complex.
            - Shape: (Ns, cs)
        taus : torch.Tensor
            The second batch of simplices corresponds to a skeleton of the complex.
            - Shape: (Nt, ct)
        q : int
            First parameter of the qij-connectivity relation.
        i : int
            Second parameter of the qij-connectivity relation. Determines the first face map of the ordered pair of face maps.
        j : int
            Third parameter of the qij-connectivity relation. Determines the second face map of the ordered pair of face maps.
        chunk_size : int, optional
            The size of the chunks to process. Default is 1024.

        Returns
        -------
        torch.Tensor
            The indices of the qij-connected simplices of the pair of skeletons.
            - Shape: (2, N)
        """

        if q > self.complex_dim:
            raise ValueError("q has to be lower than the complex dimension")

        di_sigmas = self._d_i_batched(i, sigmas)
        dj_taus = self._d_i_batched(j, taus)

        contained = self._multiple_contained_chunked(sigmas, taus, chunk_size)

        alpha_q_contained = self._alpha_q_contained_sparse(
            di_sigmas, dj_taus, q, chunk_size
        )

        return (
            torch.cat(
                (contained._indices().t(), alpha_q_contained._indices().t()),
                dim=0,
            )
            .unique(dim=0)
            .t()
        )

    def find_paths(self, indices: torch.tensor, threshold: int):
        """Find paths in adjacency matrix for (q, d_i, d_j)-connectivity relation.

        Parameters
        ----------
        indices : torch.Tensor
            The indices of the qij-connected simplices of the pair of skeletons.
            - Shape: (2, N)
        threshold : int
            The length threshold to select paths.

        Returns
        -------
        List[List]
            List of selected paths longer than the threshold.

        Notes
        -----
        This method identifies paths in the adjacency matrix associated with the
        (q, d_i, d_j)-connectivity relation. It selects paths with length
        exceeding the specified threshold.
        """

        def dfs(node, adj_list, all_paths, path):
            """
            Perform depth-first search to find paths in the adjacency matrix.

            Parameters
            ----------
            node : int
                The current node being explored.
            adj_list : List[List[int]]
                Adjacency list representation of the graph.
            all_paths : List[List[int]]
                List to store all found paths.
            path : List[int]
                The current path being explored.

            Notes
            -----
            This function recursively explores the graph using depth-first search
            to find all possible paths. It modifies the `all_paths` list in-place.

            The function does not return anything explicitly; instead, it updates
            the `all_paths` list with new paths as they are discovered.
            """
            if node not in adj_list:  # end of recursion
                if len(path) > threshold:
                    all_paths = add_path(path.copy(), all_paths)
                return

            only_loops = True
            for new_node in adj_list[node]:
                if new_node not in path:  # avoid cycles
                    only_loops = False
                    path.append(new_node)
                    dfs(new_node, adj_list, all_paths, path)
                    path.pop()

            if (
                only_loops and len(path) > threshold
            ):  # then we have another longest path
                all_paths = add_path(path.copy(), all_paths)

            return

        def edge_index_to_adj_list(edge_index):
            """Convert edge index to adjacency list representation.

            Parameters
            ----------
            edge_index : torch.Tensor
                The edge index tensor.

            Returns
            -------
            Dict[int, List[int]]
                The adjacency list representation of the graph.
            """
            adj_list = {}
            for e in edge_index.T:
                if e[0].item() not in adj_list:
                    adj_list[e[0].item()] = [e[1].item()]
                else:
                    adj_list[e[0].item()].append(e[1].item())

            return adj_list

        def is_subpath(p1, p2):
            """Check if p1 is a subpath of p2.

            Parameters
            ----------
            p1 : List[int]
                The first path.
            p2 : List[int]
                The second path.

            Returns
            -------
            bool
                True if p1 is a subpath of p2, False otherwise.
            """
            if len(p1) > len(p2):
                return False
            if len(p1) == len(p2):
                return p1 == p2
            diff = len(p2) - len(p1)
            return any(p2[i : i + len(p1)] == p1 for i in range(diff + 1))

        def add_path(new_path, all_paths):
            """Add a new path to the list of all paths.

            Parameters
            ----------
            new_path : List[int]
                The new path to add.
            all_paths : List[List[int]]
                The list of all paths.

            Returns
            -------
            List[List[int]]
                The updated list of all paths.
            """
            for path in all_paths:
                if is_subpath(new_path, path):
                    # don't add a subpath
                    return new_path
            # Check if some paths need to be removed
            for path in all_paths:
                if is_subpath(path, new_path):
                    all_paths.remove(path)
            all_paths.append(new_path)
            return all_paths

        adj_list = edge_index_to_adj_list(indices)

        all_paths = []

        for src in adj_list:
            path = [src]
            dfs(src, adj_list, all_paths, path)

        return all_paths
