"""
Plotly visualizations for the neural network project.

Two interactive charts are available:

  show_architecture(layer_sizes, activations)
      Opens a browser tab showing the network as nodes connected by edges.
      Nodes are capped at MAX_VISIBLE_NODES per layer for readability.

  show_training_progress(loss_history, val_acc_history)
      Opens a browser tab with loss and validation-accuracy curves.
"""

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

