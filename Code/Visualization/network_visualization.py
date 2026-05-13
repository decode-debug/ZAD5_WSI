"""
Plotly visualizations for the neural network project.

Three interactive charts are available:

  show_architecture(layer_sizes, activations)
      2D node-edge diagram in the browser.

  show_architecture_3d(layer_sizes, activations)
      Full 3D view — drag to orbit, scroll to zoom, right-click to pan.
      Nodes for each layer are arranged in a circle so depth is visible.

  show_training_progress(loss_history, val_acc_history)
      Side-by-side loss and val-accuracy curves.
"""

import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Maximum nodes drawn per layer (keeps the diagram readable)
MAX_VISIBLE_NODES = 8

# Colour palette per layer type
LAYER_COLORS = {
    'input':   '#636EFA',   # blue
    'relu':    '#00CC96',   # green
    'softmax': '#FFA15A',   # orange
}


# -----------------------------------------------------------------------
# Chart 1: Network Architecture
# -----------------------------------------------------------------------
def show_architecture(layer_sizes, activations):
    """
    Draw the neural network as an interactive node-edge diagram.

    Parameters
    ----------
    layer_sizes : list[int]   e.g. [64, 128, 64, 10]
    activations : list[str]   e.g. ['relu', 'relu', 'softmax']
                              (one entry per layer, NOT counting the input)
    """
    # Prepend 'input' so activations align with layer_sizes
    act_labels = ['input'] + list(activations)

    node_x, node_y, node_text, node_color = [], [], [], []
    edge_x,  edge_y  = [], []

    num_layers  = len(layer_sizes)
    x_positions = [i / (num_layers - 1) for i in range(num_layers)]

    # Collect node positions layer by layer
    layer_node_positions = []
    for col, (size, act) in enumerate(zip(layer_sizes, act_labels)):
        visible  = min(size, MAX_VISIBLE_NODES)
        y_vals   = [v / (visible - 1) if visible > 1 else 0.5
                    for v in range(visible)]
        # Centre the column vertically
        y_vals   = [y - 0.5 for y in y_vals]
        color    = LAYER_COLORS.get(act.lower(), '#EF553B')

        positions = []
        for y in y_vals:
            node_x.append(x_positions[col])
            node_y.append(y)
            node_text.append(
                f"Layer {col} ({act})<br>Size: {size}"
                + (" [truncated]" if size > MAX_VISIBLE_NODES else "")
            )
            node_color.append(color)
            positions.append((x_positions[col], y))

        layer_node_positions.append(positions)

    # Draw edges between every node in adjacent layers
    for i in range(len(layer_node_positions) - 1):
        for x1, y1 in layer_node_positions[i]:
            for x2, y2 in layer_node_positions[i + 1]:
                edge_x += [x1, x2, None]
                edge_y += [y1, y2, None]

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='rgba(150,150,150,0.25)', width=0.8),
        hoverinfo='none',
        name='connections',
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(size=14, color=node_color, line=dict(width=1, color='white')),
        text=node_text,
        hovertemplate='%{text}<extra></extra>',
        name='nodes',
    ))

    # Layer labels along the top
    for col, (size, act) in enumerate(zip(layer_sizes, act_labels)):
        fig.add_annotation(
            x=x_positions[col], y=0.62,
            text=f"<b>{act.capitalize()}</b><br>{size} nodes",
            showarrow=False,
            font=dict(size=11),
            xref='x', yref='y',
        )

    fig.update_layout(
        title='Neural Network Architecture',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#1e1e2e',
        paper_bgcolor='#1e1e2e',
        font=dict(color='white'),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig.show()


# -----------------------------------------------------------------------
# Chart 1b: 3D Network Architecture  (drag to orbit, scroll to zoom)
# -----------------------------------------------------------------------
def show_architecture_3d(layer_sizes, activations):
    """
    3D interactive neural network diagram.

    Controls (browser):
      Left-drag   → orbit / rotate
      Scroll      → zoom in / out
      Right-drag  → pan

    Layout: each layer is a ring of nodes in the Y-Z plane,
    spaced along the X axis, so rotating reveals the depth.

    Parameters
    ----------
    layer_sizes : list[int]   e.g. [64, 128, 64, 10]
    activations : list[str]   e.g. ['relu', 'relu', 'softmax']
    """
    act_labels = ['input'] + list(activations)
    num_layers = len(layer_sizes)

    # X positions: spread layers evenly along the X axis
    x_gap = 3.0
    ring_radius = 1.2     # radius of the circle of nodes inside each layer

    COLORS = {
        'input':   '#636EFA',
        'relu':    '#00CC96',
        'softmax': '#FFA15A',
    }

    node_x, node_y, node_z = [], [], []
    node_colors, node_sizes, node_labels = [], [], []
    edge_x, edge_y, edge_z = [], [], []

    # node positions per layer, needed for drawing edges
    layer_positions = []

    for col, (size, act) in enumerate(zip(layer_sizes, act_labels)):
        color  = COLORS.get(act.lower(), '#EF553B')
        x_pos  = col * x_gap

        # Scale ring radius so large layers spread out more
        radius = ring_radius * max(1.0, math.sqrt(size / 16))

        # Node marker size: shrink for dense layers so they don't overlap
        dot_size = max(3, int(10 - math.log2(max(size, 1))))

        positions = []
        for j in range(size):
            # Place all nodes evenly around a circle in the Y-Z plane
            angle = 2 * math.pi * j / size
            y = radius * math.cos(angle)
            z = radius * math.sin(angle)

            node_x.append(x_pos)
            node_y.append(y)
            node_z.append(z)
            node_colors.append(color)
            node_sizes.append(dot_size)
            node_labels.append(
                f"<b>{act.capitalize()} layer</b><br>"
                f"Layer {col} | node {j + 1}/{size}"
            )
            positions.append((x_pos, y, z))

        layer_positions.append(positions)

    # Build edges — sample up to MAX_EDGES_PER_PAIR per adjacent layer pair
    # to keep the browser fast while still showing density
    MAX_EDGES_PER_PAIR = 600
    rng = __import__('random')
    rng.seed(42)

    for i in range(len(layer_positions) - 1):
        left  = layer_positions[i]
        right = layer_positions[i + 1]
        # All possible pairs; sample if too many
        pairs = [(l, r) for l in left for r in right]
        if len(pairs) > MAX_EDGES_PER_PAIR:
            pairs = rng.sample(pairs, MAX_EDGES_PER_PAIR)
        for (x1, y1, z1), (x2, y2, z2) in pairs:
            edge_x += [x1, x2, None]
            edge_y += [y1, y2, None]
            edge_z += [z1, z2, None]

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(180,180,180,0.12)', width=0.8),
        hoverinfo='none',
        name='connections',
    ))

    # Nodes — use per-node sizes for variable-density layers
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=0),
            opacity=0.90,
        ),
        text=node_labels,
        hovertemplate='%{text}<extra></extra>',
        name='nodes',
    ))

    # Layer title labels (as annotations in 3D via invisible scatter at top)
    label_x, label_y, label_z, label_text = [], [], [], []
    for col, (size, act) in enumerate(zip(layer_sizes, act_labels)):
        r = ring_radius * max(1.0, math.sqrt(size / 16))
        label_x.append(col * x_gap)
        label_y.append(r + 0.6)
        label_z.append(0.0)
        label_text.append(f"{act.capitalize()}<br>{size} nodes")
    fig.add_trace(go.Scatter3d(
        x=label_x, y=label_y, z=label_z,
        mode='text',
        text=label_text,
        textfont=dict(size=11, color='white'),
        hoverinfo='none',
        name='labels',
    ))

    fig.update_layout(
        title=dict(text='Neural Network — 3D View', font=dict(size=20, color='white')),
        scene=dict(
            xaxis=dict(title='Layer', showgrid=False, zeroline=False,
                       backgroundcolor='#1e1e2e', gridcolor='#333'),
            yaxis=dict(title='', showgrid=False, zeroline=False,
                       backgroundcolor='#1e1e2e', gridcolor='#333',
                       showticklabels=False),
            zaxis=dict(title='', showgrid=False, zeroline=False,
                       backgroundcolor='#1e1e2e', gridcolor='#333',
                       showticklabels=False),
            bgcolor='#1e1e2e',
            # Initial camera angle — slightly elevated and rotated
            camera=dict(
                eye=dict(x=1.6, y=-1.8, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor='#1e1e2e',
        font=dict(color='white'),
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    fig.show()


# -----------------------------------------------------------------------
# Chart 2: Training Progress
# -----------------------------------------------------------------------
def show_training_progress(loss_history, val_acc_history):
    """
    Plot loss and validation accuracy curves side by side.

    Parameters
    ----------
    loss_history    : list[float]  — one value per logged checkpoint
    val_acc_history : list[float]  — one value per logged checkpoint
    """
    epochs = [i * 10 for i in range(1, len(loss_history) + 1)]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training Loss', 'Validation Accuracy'),
    )

    # Loss curve
    fig.add_trace(
        go.Scatter(
            x=epochs, y=loss_history,
            mode='lines+markers',
            line=dict(color='#EF553B', width=2),
            marker=dict(size=6),
            name='Loss',
            hovertemplate='Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>',
        ),
        row=1, col=1,
    )

    # Val accuracy curve
    fig.add_trace(
        go.Scatter(
            x=epochs, y=val_acc_history,
            mode='lines+markers',
            line=dict(color='#00CC96', width=2),
            marker=dict(size=6),
            name='Val Accuracy',
            hovertemplate='Epoch %{x}<br>Accuracy: %{y:.4f}<extra></extra>',
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text='Epoch', gridcolor='#333')
    fig.update_yaxes(gridcolor='#333')
    fig.update_layout(
        title='Training Progress',
        plot_bgcolor='#1e1e2e',
        paper_bgcolor='#1e1e2e',
        font=dict(color='white'),
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40),
    )

    fig.show()

