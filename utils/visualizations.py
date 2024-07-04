import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def figure_to_array(fig):
    """Convert a Matplotlib figure to a 3D NumPy array of RGBA values."""
    canvas = FigureCanvas(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    image_array = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8)
    width, height = fig.get_size_inches() * fig.dpi
    image_array = image_array.reshape((int(height), int(width), -1))
    return image_array

def save_figures_as_image(figures, output_filename):
    """Convert a list of figures to NumPy arrays, concatenate them horizontally, and save as a PNG image."""
    # Convert each figure to a NumPy array
    arrays = [figure_to_array(fig) for fig in figures]
    
    # Concatenate the arrays along the width
    combined_array = np.concatenate(arrays, axis=0)
    
    # Create a new figure to house the combined image
    fig = plt.figure(figsize=(combined_array.shape[1] / 100, combined_array.shape[0] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.axis('off')
    
    # Display the combined image
    #ax.imshow(combined_array,aspect='auto')
    ax.imshow(combined_array, aspect='auto', extent=[0, combined_array.shape[1], 0, combined_array.shape[0]])
    
    # Save the unified tableau to a file
    fig.savefig(output_filename)
    
    # Tenderly bid farewell to the figure, making way for new visual explorations
    plt.close(fig)


def add_text_below_figure(fig, text, fontsize=12, text_color='black'):
    """Add text below a Matplotlib figure."""
    # Compute the space needed for the text
    text_height_in_inches = fontsize * len(text.split('\n')) / 72.0

    # Adjust the figure size and plot region to accommodate the text
    fig.subplots_adjust(bottom=text_height_in_inches)
    fig.set_figheight(fig.get_figheight() + text_height_in_inches)

    # Add the text
    fig.text(0.5, 0, text, ha='center', va='top', fontsize=fontsize, color=text_color)
    return fig


def save_pickle(data, file_name, directory="."):
    """
    Saves data to a pickle file. Checks if a file with the same name already exists.

    Parameters:
    - data: The data to be pickled
    - file_name: The name of the pickle file
    - directory: The directory where the pickle file will be saved (default is the current directory)

    Returns:
    - None
    """
    
    # Create the full path for the file
    full_path = os.path.join(directory, file_name)
    
    # Check if a file with the same name already exists
    if os.path.exists(full_path):
        print(f"A file named {file_name} already exists in the directory {directory}.")
        return
    

def convert_figure_to_image(figures):
    """Convert a list of figures to NumPy arrays, concatenate them horizontally, and save as a PNG image."""
    # Convert each figure to a NumPy array
    arrays = [figure_to_array(fig) for fig in figures]
    
    # Concatenate the arrays along the width
    combined_array = np.concatenate(arrays, axis=0)
    
    # Create a new figure to house the combined image
    fig = plt.figure(figsize=(combined_array.shape[1] / 100, combined_array.shape[0] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.axis('off')
    
    # Display the combined image
    #ax.imshow(combined_array,aspect='auto')
    return combined_array,ax
    
    
    