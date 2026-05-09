"""Render Mermaid diagrams in Jupyter notebooks via mermaid.ink."""

import base64
import io

import matplotlib.pyplot as plt
import requests
from PIL import Image as im


def mm(graph: str) -> None:
    """Render a Mermaid diagram via mermaid.ink.

    Args:
        graph: Mermaid diagram syntax string (without the ```mermaid fences).

    Example:
        mm('''
        flowchart LR
            A[Start] --> B[End]
        ''')
    """
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    img = im.open(io.BytesIO(requests.get("https://mermaid.ink/img/" + base64_string).content))
    plt.figure(figsize=(10, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
