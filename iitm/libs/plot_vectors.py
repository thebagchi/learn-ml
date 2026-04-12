import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_vectors(vectors, colors=None, labels=None, xlim=(-5, 5), ylim=(-5, 5)):
    """
    Plot vectors as arrows from the origin.

    Parameters:
    vectors: list/array of 2D vectors (each can be a tensor, list, or tuple)
    colors: list of colors for each vector (optional)
    labels: list of labels for each vector (optional)
    xlim: tuple for x-axis limits
    ylim: tuple for y-axis limits
    """
    # Convert input to list if it's not already
    if not isinstance(vectors, (list, tuple)):
        vectors = [vectors]

    # Convert all vectors to tensors and validate
    tensor_vectors = []
    for i, vec in enumerate(vectors):
        if isinstance(vec, torch.Tensor):
            if vec.dim() == 1 and vec.shape[0] == 2:
                tensor_vectors.append(vec)
            elif vec.dim() == 2 and vec.shape == (1, 2):
                tensor_vectors.append(vec.squeeze(0))
            else:
                raise ValueError(f"Vector {i} must be 1D with 2 components or 2D with shape (1, 2), got {vec.shape}")
        elif isinstance(vec, (list, tuple, np.ndarray)):
            if len(vec) == 2:
                tensor_vectors.append(torch.tensor(vec, dtype=torch.float32))
            else:
                raise ValueError(f"Vector {i} must have 2 components, got {len(vec)}")
        else:
            raise ValueError(f"Vector {i} must be a tensor, list, tuple, or numpy array")

    # Stack into (N, 2) tensor
    if len(tensor_vectors) == 1:
        vectors_tensor = tensor_vectors[0].unsqueeze(0)
    else:
        vectors_tensor = torch.stack(tensor_vectors)

    # Convert to numpy for plotting
    vectors_np = vectors_tensor.numpy()

    fig, ax = plt.subplots(figsize=(8, 6))

    num_vectors = vectors_tensor.shape[0]

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, num_vectors))

    if labels is None:
        labels = [f'Vector {i+1}' for i in range(num_vectors)]

    for i in range(num_vectors):
        ax.arrow(0, 0, vectors_np[i, 0], vectors_np[i, 1],
                head_width=0.1, head_length=0.1,
                fc=colors[i], ec=colors[i],
                label=labels[i], alpha=0.7)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Vector Plot')
    ax.legend()
    plt.show()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Vector Plot')
    ax.legend()
    plt.show()