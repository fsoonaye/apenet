# apenet/eye/tree.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any, Union

def plot_decision_boundaries(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Tuple[int, int] = (0, 1),
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    resolution: int = 100,
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    show_points: bool = True
) -> plt.Figure:
    """
    Plot decision boundaries for a trained classifier on a 2D plane.
    
    Parameters
    ----------
    model : Any
        The trained model with a predict method.
    X : np.ndarray
        The input data used for training.
    y : np.ndarray
        The target labels.
    feature_indices : Tuple[int, int], default=(0, 1)
        The indices of the two features to plot.
    feature_names : Optional[List[str]], default=None
        The names of the features. If None, will use "Feature X" where X is the index.
    class_names : Optional[List[str]], default=None
        The names of the classes. If None, will use the unique values from y.
    resolution : int, default=100
        The resolution of the decision boundary grid.
    alpha : float, default=0.7
        The transparency of the scatter points.
    figsize : Tuple[int, int], default=(10, 8)
        The figure size.
    cmap : str, default="viridis"
        The colormap to use for the decision boundaries.
    show_points : bool, default=True
        Whether to show the data points.
        
    Returns
    -------
    plt.Figure
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
    tree: Any,
    max_depth: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    fontsize: int = 10,
    node_size: int = 1500,
    alpha: float = 0.7
) -> plt.Figure:
    """
    Plot the structure of a decision tree as a graph.
    
    Parameters
    ----------
    tree : Any
        The decision tree to visualize (must have a 'tree' attribute that is the root node).
    max_depth : Optional[int], default=None
        The maximum depth of the tree to visualize. If None, visualize the entire tree.
    feature_names : Optional[List[str]], default=None
        The names of the features. If None, will use "Feature X" where X is the index.
    class_names : Optional[List[str]], default=None
        The names of the classes. If None, will use the values directly.
    figsize : Tuple[int, int], default=(12, 8)
        The figure size.
    fontsize : int, default=10
        The font size for node and edge labels.
    node_size : int, default=1500
        The size of the nodes.
    alpha : float, default=0.7
        The transparency of the nodes.
        
    Returns
    -------
    plt.Figure
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
    
    # Get the root node
    root = tree.tree
    
    # Node counter for unique IDs
    node_id = 0
    
    # Process nodes recursively
    def add_node(node, parent_id=None, depth=0, edge_label=""):
        nonlocal node_id
        current_id = node_id
        node_id += 1
        
        # Skip if we reached max_depth
        if max_depth is not None and depth > max_depth:
            return
        
        # Node attributes
        if node.is_leaf_node():
            value = node.value
            if class_names is not None and 0 <= value < len(class_names):
                value_str = class_names[value]
            else:
                value_str = str(value)
            
            label = f"Class: {value_str}"
            color = "lightblue"
        else:
            feature = node.feature
            if feature_names is not None and 0 <= feature < len(feature_names):
                feature_str = feature_names[feature]
            else:
                feature_str = f"Feature {feature}"
            
            label = f"{feature_str}\n<= {node.threshold:.3f}"
            color = "lightgreen"
        
        # Add node to graph
        G.add_node(current_id, label=label, color=color, depth=depth)
        
        # Add edge from parent if not root
        if parent_id is not None:
            G.add_edge(parent_id, current_id, label=edge_label)
        
        # Recursively add children if not a leaf
        if not node.is_leaf_node():
            add_node(node.left, current_id, depth + 1, "Yes")
            add_node(node.right, current_id, depth + 1, "No")
    
    # Start with the root node
    add_node(root)
    
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
    edge_labels = {(u, v): G.edges[u, v]["label"] for u, v in G.edges}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=fontsize,
        ax=ax
    )
    
    # Remove axis
    ax.axis("off")
    
    plt.title("Decision Tree Structure")
    plt.tight_layout()
    
    return fig

def plot_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6),
    color: str = "skyblue",
    alpha: float = 0.7
) -> plt.Figure:
    """
    Plot feature importance from a tree-based model.
    
    Parameters
    ----------
    model : Any
        The trained model (must have trees attribute for RandomForest or tree attribute for DecisionTree).
    feature_names : Optional[List[str]], default=None
        The names of the features. If None, will use "Feature X" where X is the index.
    top_n : Optional[int], default=None
        The number of top features to show. If None, show all features.
    figsize : Tuple[int, int], default=(10, 6)
        The figure size.
    color : str, default="skyblue"
        The color of the bars.
    alpha : float, default=0.7
        The transparency of the bars.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    # Get feature importance
    if hasattr(model, 'trees'):  # RandomForest
        # Combine importances from all trees
        importance = {}
        for tree in model.trees:
            tree_importance = tree.feature_importance()
            for feature, count in tree_importance.items():
                importance[feature] = importance.get(feature, 0) + count
    elif hasattr(model, 'tree'):  # DecisionTree
        importance = model.feature_importance()
    else:
        raise ValueError("Model must be a DecisionTree or RandomForest with appropriate attributes")
    
    # Convert to DataFrame for easier handling
    import pandas as pd
    df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False)
    
    # Limit to top_n if specified
    if top_n is not None:
        df = df.head(top_n)
    
    # Map feature indices to names if provided
    if feature_names is not None:
        df['Feature'] = df['Feature'].apply(
            lambda x: feature_names[x] if x < len(feature_names) else f"Feature {x}"
        )
    else:
        df['Feature'] = df['Feature'].apply(lambda x: f"Feature {x}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        df['Feature'],
        df['Importance'],
        color=color,
        alpha=alpha
    )
    
    ax.set_title('Feature Importance')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance (Number of splits)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def plot_decision_regions(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Tuple[int, int] = (0, 1),
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    resolution: int = 100,
    show_tree_splits: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot decision regions with tree splits for a decision tree model on a 2D plane.
    
    Parameters
    ----------
    model : Any
        The trained decision tree model.
    X : np.ndarray
        The input data used for training.
    y : np.ndarray
        The target labels.
    feature_indices : Tuple[int, int], default=(0, 1)
        The indices of the two features to plot.
    feature_names : Optional[List[str]], default=None
        The names of the features. If None, will use "Feature X" where X is the index.
    class_names : Optional[List[str]], default=None
        The names of the classes. If None, will use the unique values from y.
    resolution : int, default=100
        The resolution of the decision boundary grid.
    show_tree_splits : bool, default=True
        Whether to show the actual decision tree splits.
    figsize : Tuple[int, int], default=(10, 8)
        The figure size.
        
    Returns
    -------
    plt.Figure
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
    if show_tree_splits and hasattr(model, 'tree'):
        # Find all the splits involving our two features
        splits = []
        
        def traverse_tree(node, depth=0):
            if node.is_leaf_node():
                return
            
            # Check if this node splits on one of our features
            if node.feature == feature_indices[0]:
                # This is a vertical line at x = threshold
                splits.append(('v', node.threshold))
            elif node.feature == feature_indices[1]:
                # This is a horizontal line at y = threshold
                splits.append(('h', node.threshold))
            
            # Continue traversing
            traverse_tree(node.left, depth + 1)
            traverse_tree(node.right, depth + 1)
        
        # Start traversing from the root
        traverse_tree(model.tree)
        
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
            
            # Define the traversal function
            def traverse_tree(node, depth=0):
                if node.is_leaf_node():
                    return
                
                # Check if this node splits on one of our features
                if node.feature == feature_indices[0]:
                    # This is a vertical line at x = threshold
                    ax.axvline(
                        x=node.threshold,
                        color=color,
                        linestyle='--',
                        alpha=0.3,
                        linewidth=0.8
                    )
                elif node.feature == feature_indices[1]:
                    # This is a horizontal line at y = threshold
                    ax.axhline(
                        y=node.threshold,
                        color=color,
                        linestyle='--',
                        alpha=0.3,
                        linewidth=0.8
                    )
                
                # Continue traversing
                traverse_tree(node.left, depth + 1)
                traverse_tree(node.right, depth + 1)
            
            # Start traversing from the root
            traverse_tree(tree.tree)
    
    plt.title(f"Decision Regions with Tree Splits for {feature_names[0]} vs {feature_names[1]}")
    plt.tight_layout()
    
    return fig

def plot_tree_decisions_3d(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Tuple[int, int, int] = (0, 1, 2),
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    resolution: int = 20,
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "viridis",
    elev: int = 30,
    azim: int = 30
) -> plt.Figure:
    """
    Plot decision boundaries for a trained classifier in 3D space.
    
    Parameters
    ----------
    model : Any
        The trained model with a predict method.
    X : np.ndarray
        The input data used for training.
    y : np.ndarray
        The target labels.
    feature_indices : Tuple[int, int, int], default=(0, 1, 2)
        The indices of the three features to plot.
    feature_names : Optional[List[str]], default=None
        The names of the features. If None, will use "Feature X" where X is the index.
    class_names : Optional[List[str]], default=None
        The names of the classes. If None, will use the unique values from y.
    resolution : int, default=20
        The resolution of the decision boundary grid (lower for 3D to avoid memory issues).
    alpha : float, default=0.7
        The transparency of the scatter points.
    figsize : Tuple[int, int], default=(12, 10)
        The figure size.
    cmap : str, default="viridis"
        The colormap to use for the decision boundaries.
    elev : int, default=30
        The elevation angle for 3D view.
    azim : int, default=30
        The azimuth angle for 3D view.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract the three features we want to visualize
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
    z_min, z_max = X_feat[:, 2].min() - 0.1, X_feat[:, 2].max() + 0.1
    
    xx, yy, zz = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
        np.linspace(z_min, z_max, resolution)
    )
    
    # Create the full feature vector for prediction
    if len(X.shape) > 1:
        n_features = X.shape[1]
    else:
        n_features = 1
    
    # Flatten the meshgrid
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    
    # Create full feature matrix
    X_full = np.zeros((grid_points.shape[0], n_features))
    
    # Fill in the three features we're interested in
    X_full[:, feature_indices[0]] = grid_points[:, 0]
    X_full[:, feature_indices[1]] = grid_points[:, 1]
    X_full[:, feature_indices[2]] = grid_points[:, 2]
    
    # Predict on the grid
    Z = model.predict(X_full)
    
    # Create the plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set the view angles
    ax.view_init(elev=elev, azim=azim)
    
    # Scatter plot for each class
    custom_cmap = plt.get_cmap(cmap, len(unique_classes))
    for i, cls in enumerate(unique_classes):
        idx = y == cls
        ax.scatter(
            X_feat[idx, 0],
            X_feat[idx, 1],
            X_feat[idx, 2],
            c=[custom_cmap(i / (len(unique_classes) - 1))],
            label=class_names[i],
            alpha=alpha,
            edgecolors='k'
        )
    
    # Visualize the decision boundaries by plotting points
    # (we use a sample of the grid to avoid plotting too many points)
    sample_size = min(10000, grid_points.shape[0])
    sample_idx = np.random.choice(grid_points.shape[0], sample_size, replace=False)
    
    for i, cls in enumerate(unique_classes):
        idx = (Z == cls) & np.isin(np.arange(len(Z)), sample_idx)
        if np.any(idx):
            ax.scatter(
                grid_points[idx, 0],
                grid_points[idx, 1],
                grid_points[idx, 2],
                c=[custom_cmap(i / (len(unique_classes) - 1))],
                marker='.',
                alpha=0.1,
                s=10
            )
    
    # Add labels
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    ax.set_title('3D Decision Boundaries')
    ax.legend()
    
    plt.tight_layout()
    return fig