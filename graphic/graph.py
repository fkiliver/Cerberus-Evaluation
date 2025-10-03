import networkx as nx
from networkx.drawing.nx_agd import to_agraph
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns

class AttackChainVisualizer:
    """A class for visualizing attack chains using directed graphs."""
    
    def __init__(self, 
                 dpi: int = 300,
                 font_size: int = 10,
                 edge_font_size: int = 8):
        """
        Initialize the visualizer.
        
        Args:
            dpi: DPI for the output image
            font_size: Font size for node labels
            edge_font_size: Font size for edge labels
        """
        self.dpi = dpi
        self.font_size = font_size
        self.edge_font_size = edge_font_size
        self._setup_style()
        
    def _setup_style(self) -> None:
        """Set up the visualization style."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def _create_graph(self) -> nx.DiGraph:
        """
        Create the attack chain graph.
        
        Returns:
            Directed graph representing the attack chain
        """
        G = nx.DiGraph()
        
        # Define nodes with types
        nodes = [
            ("Web Browser", {"type": "process", "label": "Web Browser\n(Process)"}),
            ("PowerShell", {"type": "process", "label": "PowerShell\n(Process)"}),
            ("malicious_script.ps1", {"type": "file", "label": "malicious_script.ps1\n(File)"}),
            ("backdoor.exe", {"type": "file", "label": "backdoor.exe\n(File)"}),
            ("192.168.1.100:443", {"type": "network", "label": "192.168.1.100:443\n(Network)"}),
            ("sensitive_data.txt", {"type": "file", "label": "sensitive_data.txt\n(File)"}),
            ("exfiltrated_data.enc", {"type": "file", "label": "exfiltrated_data.enc\n(File)"})
        ]
        
        # Add nodes
        for node, attrs in nodes:
            G.add_node(node, **attrs)
        
        # Define edges
        edges = [
            ("Web Browser", "PowerShell", "launches"),
            ("PowerShell", "malicious_script.ps1", "executes"),
            ("malicious_script.ps1", "backdoor.exe", "creates"),
            ("PowerShell", "backdoor.exe", "executes"),
            ("backdoor.exe", "192.168.1.100:443", "connects to"),
            ("backdoor.exe", "sensitive_data.txt", "reads"),
            ("backdoor.exe", "exfiltrated_data.enc", "writes"),
            ("exfiltrated_data.enc", "192.168.1.100:443", "transfers")
        ]
        
        # Add edges
        for src, dst, label in edges:
            G.add_edge(src, dst, label=label)
            
        return G
    
    def _get_node_styles(self) -> Dict[str, Dict[str, str]]:
        """
        Get the style configuration for different node types.
        
        Returns:
            Dictionary mapping node types to their style attributes
        """
        return {
            "process": {"color": "#E1F5FE", "style": "filled", "shape": "box"},
            "file": {"color": "#E8F5E9", "style": "filled", "shape": "ellipse"},
            "network": {"color": "#FFEBEE", "style": "filled", "shape": "diamond"}
        }
    
    def _apply_graph_attributes(self, A: 'AGraph', G: nx.DiGraph) -> None:
        """
        Apply attributes to the graph visualization.
        
        Args:
            A: AGraph object for visualization
            G: NetworkX graph with data
        """
        # Set graph attributes
        A.graph_attr.update(
            dpi=str(self.dpi),
            rankdir="LR",  # Left to right layout
            splines="ortho"
        )
        
        # Set default node attributes
        A.node_attr.update(
            fontname="Helvetica",
            fontsize=str(self.font_size)
        )
        
        # Apply node styles
        node_styles = self._get_node_styles()
        for node in A.nodes():
            node_type = G.nodes[node.get_name()]["type"]
            node.attr.update(**node_styles[node_type])
            node.attr.update(label=G.nodes[node.get_name()]["label"])
        
        # Set edge attributes
        for edge in A.edges():
            edge.attr.update(
                fontname="Helvetica",
                fontsize=str(self.edge_font_size),
                fontcolor="#616161",
                arrowsize="0.5"
            )
            edge.attr["label"] = G.edges[edge].get("label", "")
    
    def visualize_attack_chain(self, output_path: str = "attack_chain.png") -> None:
        """
        Create and save the attack chain visualization.
        
        Args:
            output_path: Path to save the output image
        """
        # Create graph
        G = self._create_graph()
        
        # Convert to AGraph
        A = to_agraph(G)
        
        # Apply attributes
        self._apply_graph_attributes(A, G)
        
        # Save visualization
        A.draw(output_path, format="png", prog="dot")
        print(f"Attack chain visualization saved as {output_path}")

def main():
    """Example usage of the AttackChainVisualizer."""
    visualizer = AttackChainVisualizer()
    visualizer.visualize_attack_chain()

if __name__ == "__main__":
    main()