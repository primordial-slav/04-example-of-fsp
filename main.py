import cv2
import numpy as np
from decoding import ImageProcessor
from dataclasses import dataclass

@dataclass
class config:
    save_figures: bool = True
    plot_figures: bool = True
if __name__ == '__main__':
    # had to do some magic here that fixes is to how we worked with it within the whole pipeline
    img = cv2.imread("examples/example_cropped.png")[...,::-1]
    img = 255. - img
    decoder = ImageProcessor(True)
    decoder.process_image("example_cropped.png", 
                          img, 
                          save_figures=config.save_figures, 
                          plot_figures=config.plot_figures)
