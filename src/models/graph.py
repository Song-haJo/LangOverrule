"""
GraphGPT Model Wrapper for Text Dominance Analysis in Graph Tasks.

Supports GraphGPT and similar graph-language models.
"""

import torch
from typing import Dict, Optional, Any, Union, List, Tuple
import numpy as np

from .base import BaseMLLMWrapper, ModelConfig
from ..attention import TokenMasks


class GraphGPTWrapper(BaseMLLMWrapper):
    """
    Wrapper for GraphGPT models.

    GraphGPT uses instruction tuning to align graph knowledge
    with large language models.
    """

    GRAPH_TOKEN = "<graph>"
    NODE_TOKEN = "<node>"
    EDGE_TOKEN = "<edge>"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_path: str = "GraphGPT-7B",  # Placeholder - adjust to actual path
    ):
        if config is None:
            config = ModelConfig(
                model_name="graphgpt",
                model_path=model_path,
            )
        super().__init__(config)
        self.model_path = model_path or config.model_path
        self.graph_token_id = None

    def load_model(self) -> None:
        """Load GraphGPT model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers>=4.40.0"
            )

        # Note: GraphGPT may require custom model loading
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.config.get_torch_dtype(),
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # Get graph token ID
        if hasattr(self.model.config, 'graph_token_index'):
            self.graph_token_id = self.model.config.graph_token_index

        self._loaded = True

    def encode_graph(
        self,
        edge_index: Union[np.ndarray, torch.Tensor],
        node_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
        num_nodes: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode graph structure for the model.

        Args:
            edge_index: Edge indices (2, E) array
            node_features: Optional node feature matrix (N, D)
            num_nodes: Number of nodes (inferred if not provided)

        Returns:
            Dictionary with encoded graph tensors
        """
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index).long()

        if node_features is not None:
            if isinstance(node_features, np.ndarray):
                node_features = torch.from_numpy(node_features).float()
        else:
            # Create identity features
            if num_nodes is None:
                num_nodes = edge_index.max().item() + 1
            node_features = torch.eye(num_nodes)

        return {
            'edge_index': edge_index,
            'node_features': node_features,
        }

    def graph_to_text(
        self,
        edge_index: Union[np.ndarray, torch.Tensor],
        node_labels: Optional[List[str]] = None,
    ) -> str:
        """
        Convert graph to text representation.

        Some graph-LLM models use textual graph description.

        Args:
            edge_index: Edge indices (2, E)
            node_labels: Optional labels for nodes

        Returns:
            Text description of the graph
        """
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.numpy()

        num_nodes = edge_index.max() + 1
        num_edges = edge_index.shape[1]

        if node_labels is None:
            node_labels = [f"node_{i}" for i in range(num_nodes)]

        lines = [f"Graph with {num_nodes} nodes and {num_edges} edges:"]
        lines.append("Edges:")
        for i in range(num_edges):
            src, dst = edge_index[0, i], edge_index[1, i]
            lines.append(f"  {node_labels[src]} -> {node_labels[dst]}")

        return "\n".join(lines)

    def preprocess(
        self,
        text: str,
        media: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess text and graph data.

        Args:
            text: Text prompt/question about the graph
            media: Dictionary with 'edge_index' and optional 'node_features'
            **kwargs: Additional arguments

        Returns:
            Dictionary with model inputs
        """
        graph_encoded = None

        if media is not None:
            if isinstance(media, dict) and 'edge_index' in media:
                graph_encoded = self.encode_graph(
                    media['edge_index'],
                    media.get('node_features'),
                    media.get('num_nodes'),
                )

        # Add graph token if not present
        if graph_encoded is not None and self.GRAPH_TOKEN not in text:
            text = f"{self.GRAPH_TOKEN}\n{text}"

        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        )

        # Add graph data
        if graph_encoded is not None:
            inputs.update(graph_encoded)

        # Move to device
        device = self.config.device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        return inputs

    def get_token_masks(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> TokenMasks:
        """Create token masks for GraphGPT."""
        if input_ids.dim() > 1:
            input_ids = input_ids[0]

        device = input_ids.device
        seq_len = len(input_ids)

        # Graph tokens
        if self.graph_token_id is not None:
            nontext_mask = (input_ids == self.graph_token_id)
        else:
            # Estimate based on node features if available
            if 'node_features' in kwargs:
                num_graph_tokens = kwargs['node_features'].shape[0]
                nontext_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
                # Approximate: assume graph tokens at specific positions
            else:
                nontext_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

        # Special tokens
        special_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if self.tokenizer is not None:
            for attr in ['pad_token_id', 'bos_token_id', 'eos_token_id']:
                token_id = getattr(self.tokenizer, attr, None)
                if token_id is not None:
                    special_mask |= (input_ids == token_id)

        text_mask = ~nontext_mask & ~special_mask

        return TokenMasks(
            text_mask=text_mask,
            nontext_mask=nontext_mask,
        )


class GraphGPTAnalyzer:
    """High-level analyzer for GraphGPT models."""

    def __init__(
        self,
        model_path: str = "GraphGPT-7B",
        device: str = "cuda",
    ):
        config = ModelConfig(
            model_name="graphgpt",
            model_path=model_path,
            device=device,
        )
        self.wrapper = GraphGPTWrapper(config)

    def analyze_graph(
        self,
        edge_index: np.ndarray,
        question: str,
        node_features: Optional[np.ndarray] = None,
        replicate_factor: int = 1,
    ) -> Dict[str, Any]:
        """
        Analyze graph data with optional token replication.

        The paper shows that graph modality initially shows non-text
        dominance (MDI < 1), but with replication, text dominance emerges.

        Args:
            edge_index: Edge indices (2, E)
            question: Question about the graph
            node_features: Optional node features
            replicate_factor: Factor to replicate graph tokens

        Returns:
            Analysis results
        """
        if not self.wrapper._loaded:
            self.wrapper.load_model()

        from ..metrics import compute_modality_metrics

        # Replicate graph if needed
        if replicate_factor > 1:
            # Replicate by creating multiple copies of the graph structure
            original_edge_index = edge_index.copy()
            num_nodes = edge_index.max() + 1
            replicated_edges = [original_edge_index]

            for i in range(1, replicate_factor):
                offset = i * num_nodes
                shifted_edges = original_edge_index + offset
                replicated_edges.append(shifted_edges)

            edge_index = np.concatenate(replicated_edges, axis=1)

            if node_features is not None:
                node_features = np.tile(node_features, (replicate_factor, 1))

        # Prepare graph data
        graph_data = {
            'edge_index': edge_index,
            'node_features': node_features,
        }

        # Preprocess
        inputs = self.wrapper.preprocess(question, graph_data)

        # Enable attention output
        self.wrapper.model.config.output_attentions = True

        # Forward pass
        with torch.no_grad():
            outputs = self.wrapper.model(**inputs, output_attentions=True)

        # Get masks and attention
        token_masks = self.wrapper.get_token_masks(inputs['input_ids'], **inputs)

        attentions = []
        if hasattr(outputs, 'attentions') and outputs.attentions:
            attentions = list(outputs.attentions)

        # Compute metrics
        metrics = {}
        if attentions:
            metrics = compute_modality_metrics(
                attentions,
                token_masks.text_mask,
                token_masks.nontext_mask,
                layerwise=True,
            )

        return {
            'metrics': metrics,
            'replicate_factor': replicate_factor,
            'num_graph_tokens': token_masks.num_nontext_tokens,
            'num_text_tokens': token_masks.num_text_tokens,
        }

    def generate_synthetic_graph(
        self,
        num_nodes: int = 10,
        edge_prob: float = 0.3,
        graph_type: str = "random",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic graph for testing.

        Args:
            num_nodes: Number of nodes
            edge_prob: Probability of edge (for random graph)
            graph_type: Type of graph ('random', 'chain', 'star', 'complete')

        Returns:
            Tuple of (edge_index, node_features)
        """
        if graph_type == "random":
            # Erdos-Renyi random graph
            edges = []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if np.random.rand() < edge_prob:
                        edges.append([i, j])
                        edges.append([j, i])  # Undirected
            edge_index = np.array(edges).T if edges else np.zeros((2, 0), dtype=int)

        elif graph_type == "chain":
            edges = []
            for i in range(num_nodes - 1):
                edges.append([i, i + 1])
                edges.append([i + 1, i])
            edge_index = np.array(edges).T

        elif graph_type == "star":
            edges = []
            for i in range(1, num_nodes):
                edges.append([0, i])
                edges.append([i, 0])
            edge_index = np.array(edges).T

        elif graph_type == "complete":
            edges = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edges.append([i, j])
            edge_index = np.array(edges).T

        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        # Simple node features
        node_features = np.eye(num_nodes)

        return edge_index, node_features
