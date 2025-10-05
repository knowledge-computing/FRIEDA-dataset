import matplotlib.pyplot as plt
import numpy as np

csfont = {'fontname':'Times New Roman'}
plt.rcParams["font.family"] = "Times New Roman"

# Set the font family to 'serif'
plt.rcParams['font.family'] = 'serif'

# Specify the preferred serif font, corresponding to Times New Roman
# Note: Matplotlib will use a Times-like font if available and configured in your LaTeX installation.
plt.rcParams['font.serif'] = ['Times New Roman']

# Set font sizes to match ICLR's standard text size (10pt)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

categories = ['A', 'B', 'C', 'D']
segment1_values = [10, 15, 7, 12]
segment2_values = [5, 8, 10, 6]
segment3_values = [3, 4, 6, 9]

# Plot the first segment
plt.bar(categories, segment1_values, label='Segment 1', color='skyblue')

# Plot the second segment on top of the first
plt.bar(categories, segment2_values, bottom=segment1_values, label='Segment 2', color='lightcoral')

# Plot the third segment on top of the first two
bottom_for_segment3 = np.array(segment1_values) + np.array(segment2_values)
plt.bar(categories, segment3_values, bottom=bottom_for_segment3, label='Segment 3', color='lightgreen')

plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Stacked Bar Chart Example')
plt.legend()
plt.savefig('./dummy.png')