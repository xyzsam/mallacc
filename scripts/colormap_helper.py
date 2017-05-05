import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_colormap(cmap, num_colors):
  """ Returns a pyplot color list with num_colors colors from a colormap.

  Args:
    cmap: a colormap that is a member of pyplot.cm.
    num_colors: The number of colors from this colormap to return.

  Returns:
    A list of color values from this colormap between 0 and 1.
  """
  return cmap(np.linspace(0, 1, num_colors))
