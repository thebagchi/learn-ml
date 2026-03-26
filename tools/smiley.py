"""
Script to create a smiley face image using matplotlib.
Generates a smiley.png file.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_smiley():
    """Create a smiley face and save as PNG."""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw face (large circle)
    face = patches.Circle((0, 0), 1, linewidth=3, edgecolor='black', facecolor='yellow')
    ax.add_patch(face)
    
    # Draw left eye
    left_eye = patches.Circle((-0.35, 0.35), 0.1, linewidth=2, edgecolor='black', facecolor='black')
    ax.add_patch(left_eye)
    
    # Draw right eye
    right_eye = patches.Circle((0.35, 0.35), 0.1, linewidth=2, edgecolor='black', facecolor='black')
    ax.add_patch(right_eye)
    
    # Draw smile (arc)
    theta = np.linspace(0, np.pi, 100)
    radius = 0.4
    x = radius * np.cos(theta)
    y = -0.2 - 0.3 * np.sin(theta)
    ax.plot(x, y, 'k-', linewidth=3)
    
    # Save the figure
    plt.savefig('res/smiley.png', dpi=100, bbox_inches='tight', facecolor='white')
    print("✓ Smiley face created and saved as 'res/smiley.png'")
    plt.close()

if __name__ == '__main__':
    create_smiley()
