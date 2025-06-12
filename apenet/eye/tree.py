# apenet/eye/tree.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_decision_boundaries(
    model,
    X,
    y,
    feature_indices=(0, 1),
    feature_names=None,
    class_names=None,
    resolution=100,
    alpha=0.7,
    figsize=(10, 8),
    cmap="viridis",
    show_points=True
):
    """Plot decision boundaries for a trained classifier on a 2D plane.
    
    Parameters
    ----------
    model : object
        The trained model with a predict method.
    X : ndarray
        The input data used for training.
    y : ndarray
        The target labels.
    feature_indices : tuple, default=(0, 1)
        The indices of the two features to plot.
    feature_names : list, default=None
        The names of the features. If None, will use "Feature X" where X is the index.
    class_names : list, default=None
        The names of the classes. If None, will use the unique values from y.
    resolution : int, default=100
        The resolution of the decision boundary grid.
    alpha : float, default=0.7
        The transparency of the scatter points.
    figsize : tuple, default=(10, 8)
        The figure size.
    cmap : str, default="viridis"
        The colormap to use for the decision boundaries.
    show_points : bool, default=True
        Whether to show the data points.
        
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure.
    """
    # Extract the two features we want to visualize
    X_feat = X[:, feature_indices]
    
    # Setup feature names
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in feature_indices]
    
    # Setup class names
    unique_classes = np.unique(y)
    if class_names is None:
        class_names = [str(cls) for cls in unique_classes]
    
    # Create a meshgrid to visualize the decision boundaries
    x_min, x_max = X_feat[:, 0].min() - 0.1, X_feat[:, 0].max() + 0.1
    y_min, y_max = X_feat[:, 1].min() - 0.1, X_feat[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Create the full feature vector for prediction
    # Start with zeros for all samples and features
    if len(X.shape) > 1:
        n_features = X.shape[1]
    else:
        n_features = 1
    
    X_full = np.zeros((resolution * resolution, n_features))
    
    # Fill in the two features we're interested in
    X_full[:, feature_indices[0]] = xx.ravel()
    X_full[:, feature_indices[1]] = yy.ravel()
    
    # Predict on the grid
    Z = model.predict(X_full)
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the decision boundary
    custom_cmap = plt.get_cmap(cmap, len(unique_classes))
    contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap=custom_cmap, levels=np.arange(len(unique_classes) + 1) - 0.5)
    
    # Add a colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_ticks(np.arange(len(unique_classes)))
    cbar.set_ticklabels(class_names)
    
    # Plot the training points if requested
    if show_points:
        scatter_colors = custom_cmap(np.linspace(0, 1, len(unique_classes)))
        for i, cls in enumerate(unique_classes):
            idx = y == cls
            ax.scatter(
                X_feat[idx, 0],
                X_feat[idx, 1],
                c=[scatter_colors[i]],
                label=class_names[i],
                edgecolors='k',
                alpha=alpha
            )
    
    # Add labels and legend
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title('Decision Boundaries')
    ax.legend()
    
    return fig

def plot_tree_structure(
    tree,
    max_depth=None,
    feature_names=None,
    class_names=None,
    figsize=(12, 8),
    fontsize=10,
    node_size=1500,
    alpha=0.7
):
    """Plot the structure of a decision tree as a graph.
    
    Parameters
    ----------
    tree : object
        The decision tree to visualize (must have a 'tree_' attribute that is the root node).
    max_depth : int, default=None
        The maximum depth of the tree to visualize. If None, visualize the entire tree.
    feature_names : list, default=None
        The names of the features. If None, will use "Feature X" where X is the index.
    class_names : list, default=None
        The names of the classes. If None, will use the values directly.
    figsize : tuple, default=(12, 8)
        The figure size.
    fontsize : int, default=10
        The font size for node and edge labels.
    node_size : int, default=1500
        The size of the nodes.
    alpha : float, default=0.7
        The transparency of the nodes.
        
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure.
    
    Notes
    -----
    This function requires networkx to be installed.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("This function requires networkx to be installed. "
                          "Please install it with 'pip install networkx'")
    
    # Create a graph
    G = nx.DiGraph()
    
    # Node counter for unique IDs
    node_id = 0
    node_map = {}  # Maps Node objects to node IDs
    
    # Create operation to build the graph
    def build_graph_operation(node, depth=0):
        nonlocal node_id
        
        # Skip if we reached max_depth
        if max_depth is not None and depth > max_depth:
            return {}
            
        current_id = node_id
        node_map[node] = current_id
        node_id += 1
        
        # Node attributes
        if node.is_leaf():
            value = node.value
            if class_names is not None and 0 <= value < len(class_names):
                value_str = class_names[value]
            else:
                value_str = str(value)
            
            label = f"Class: {value_str}"
            color = "lightblue"
        else:
            feature = node.feature_idx
            if feature_names is not None and 0 <= feature < len(feature_names):
                feature_str = feature_names[feature]
            else:
                feature_str = f"Feature {feature}"
            
            label = f"{feature_str}\n<= {node.threshold:.3f}"
            color = "lightgreen"
        
        # Add node to graph
        G.add_node(current_id, label=label, color=color, depth=depth)
        
        # We don't add edges here, but in a second pass after all nodes are created
        return {}
    
    # Traverse the tree to create all nodes
    tree._traverse_tree(tree.tree_, build_graph_operation)
    
    # Second pass to add edges
    def add_edges_operation(node, depth=0):
        if max_depth is not None and depth > max_depth:
            return {}
            
        current_id = node_map[node]
        
        # Add edges to children if not a leaf
        if not node.is_leaf():
            if node.left in node_map:
                G.add_edge(current_id, node_map[node.left], label="Yes")
            if node.right in node_map:
                G.add_edge(current_id, node_map[node.right], label="No")
        
        return {}
    
    # Traverse again to add edges
    tree._traverse_tree(tree.tree_, add_edges_operation)
    
    # Create positions for nodes using a hierarchical layout
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get node colors
    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        alpha=alpha,
        node_size=node_size,
        ax=ax
    )
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, ax=ax)
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos,
        labels={n: G.nodes[n]["label"] for n in G.nodes},
        font_size=fontsize,
        ax=ax
    )
    
    # Draw edge labels
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True) if "label" in d}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=fontsize-1,
        ax=ax
    )
    
    # Remove axis
    ax.set_axis_off()
    
    plt.tight_layout()
    return fig


def plot_decision_regions(
    model,
    X,
    y,
    feature_indices=(0, 1),
    feature_names=None,
    class_names=None,
    resolution=100,
    show_tree_splits=True,
    figsize=(10, 8)
):
    """Plot decision regions with tree splits for a decision tree model on a 2D plane.
    
    Parameters
    ----------
    model : object
        The trained decision tree model.
    X : ndarray
        The input data used for training.
    y : ndarray
        The target labels.
    feature_indices : tuple, default=(0, 1)
        The indices of the two features to plot.
    feature_names : list, default=None
        The names of the features. If None, will use "Feature X" where X is the index.
    class_names : list, default=None
        The names of the classes. If None, will use the unique values from y.
    resolution : int, default=100
        The resolution of the decision boundary grid.
    show_tree_splits : bool, default=True
        Whether to show the actual decision tree splits.
    figsize : tuple, default=(10, 8)
        The figure size.
        
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure.
    """
    # Extract the two features we want to visualize
    X_feat = X[:, feature_indices]
    
    # Setup feature names
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in feature_indices]
    
    # Setup class names
    unique_classes = np.unique(y)
    if class_names is None:
        class_names = [str(cls) for cls in unique_classes]
    
    # First plot the decision regions
    fig = plot_decision_boundaries(
        model, X, y,
        feature_indices=feature_indices,
        feature_names=feature_names,
        class_names=class_names,
        resolution=resolution,
        figsize=figsize
    )
    
    ax = fig.axes[0]
    
    # If it's a decision tree and we should show the splits
    if show_tree_splits and hasattr(model, 'tree_'):
        # Find all the splits involving our two features
        splits = []
        
        def find_splits_operation(node, depth=0):
            if node.is_leaf():
                return {}
            
            # Check if this node splits on one of our features
            if node.feature_idx == feature_indices[0]:
                # This is a vertical line at x = threshold
                splits.append(('v', node.threshold))
            elif node.feature_idx == feature_indices[1]:
                # This is a horizontal line at y = threshold
                splits.append(('h', node.threshold))
            
            return {}
        
        # Use the tree's traversal method
        model._traverse_tree(model.tree_, find_splits_operation)
        
        # Get the axis limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Plot all the splits as lines
        for direction, threshold in splits:
            if direction == 'v':  # Vertical line
                ax.axvline(x=threshold, color='red', linestyle='--', alpha=0.5)
            else:  # Horizontal line
                ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
    
    # If it's a random forest, try to plot splits from a sample of trees
    elif show_tree_splits and hasattr(model, 'trees'):
        # Sample a few trees to avoid cluttering
        num_trees_to_show = min(5, len(model.trees))
        sample_indices = np.random.choice(
            len(model.trees), 
            num_trees_to_show, 
            replace=False
        )
        trees_to_show = [model.trees[i] for i in sample_indices]
        
        # Collect splits from each tree
        for tree_idx, tree in enumerate(trees_to_show):
            color = plt.cm.tab10(tree_idx % 10)
            
            # Define the operation for finding splits
            def find_splits_operation(node, depth=0):
                if node.is_leaf():
                    return {}
                
                # Check if this node splits on one of our features
                if node.feature_idx == feature_indices[0]:
                    # This is a vertical line at x = threshold
                    ax.axvline(
                        x=node.threshold,
                        color=color,
                        linestyle='--',
                        alpha=0.3,
                        linewidth=0.8
                    )
                elif node.feature_idx == feature_indices[1]:
                    # This is a horizontal line at y = threshold
                    ax.axhline(
                        y=node.threshold,
                        color=color,
                        linestyle='--',
                        alpha=0.3,
                        linewidth=0.8
                    )
                
                return {}
            
            tree._traverse_tree(tree.tree_, find_splits_operation)
    
    plt.title(f"Decision Regions with Tree Splits for {feature_names[0]} vs {feature_names[1]}")
    plt.tight_layout()
    
    return fig