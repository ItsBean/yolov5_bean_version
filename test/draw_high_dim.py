import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Generate random 10-dimensional vectors (for demonstration)
data = np.random.rand(10, 10)


# Use these functions to map each dimension to a visual property
def map_to_shape(val):
    shapes = ['circle', 'square', 'triangle']
    idx = int(val * len(shapes))
    return shapes[idx % len(shapes)]


def map_to_color(val):
    return (val, 0.5, 0.5)


def map_to_size(val):
    return 100 + 900 * val


def map_to_texture(val):
    textures = [None, '//', '\\\\', '||']
    idx = int(val * len(textures))
    return textures[idx % len(textures)]


def map_to_edgecolor(val):
    return (0.5, val, 0.5)


def map_to_edgethickness(val):
    return 0.5 + 2.5 * val


def map_to_opacity(val):
    return 0.1 + 0.9 * val


fig, ax = plt.subplots()

for point in data:
    x, y = point[3], point[4]  # map x4 and x5 to scatter plot coordinates

    shape = map_to_shape(point[0])
    facecolor = map_to_color(point[1])
    size = map_to_size(point[2])
    texture = map_to_texture(point[5])
    edgecolor = map_to_edgecolor(point[6])
    edgethickness = map_to_edgethickness(point[7])
    opacity = map_to_opacity(point[8])
    annotation = str(round(point[9], 2))

    if shape == 'circle':
        ax.add_patch(
            patches.Circle((x, y), size, fc=facecolor, hatch=texture, ec=edgecolor, lw=edgethickness, alpha=opacity))
    elif shape == 'square':
        ax.add_patch(
            patches.Rectangle((x - size / 2, y - size / 2), size, size, fc=facecolor, hatch=texture, ec=edgecolor,
                              lw=edgethickness, alpha=opacity))
    elif shape == 'triangle':
        ax.add_patch(
            patches.RegularPolygon((x, y), numVertices=3, radius=size, fc=facecolor, hatch=texture, ec=edgecolor,
                                   lw=edgethickness, alpha=opacity))

    ax.annotate(annotation, (x, y), color='black', weight='bold', ha='center', va='center')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.show()
