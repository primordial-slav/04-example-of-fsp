import logging
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.visualizations import *
from utils.image_preprocessing import *
from utils.contour_analysis import *
from utils.curve_fitting import *
import traceback
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import Counter
import random



class ImageProcessor:
    def __init__(self,full_traceback=False,compare_with_previous=False,outputs_path="./outputs/"):
        self.CODE_DICT = {}
        self.full_traceback = full_traceback
        self.compare_with_previous = compare_with_previous
        self.outputs_path = outputs_path
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.CRITICAL)
        

    def process_crop(self, crop, fname, j):
        crop_uint8 = (crop * 255).astype('uint8')
        #_, _, msg = check_anomaly_from_decodebarcodev2(crop_uint8, self.reader)
        modified_mask = self.preprocess_crop(crop_uint8)
        self.final_image = np.zeros((crop_uint8.shape[0], crop_uint8.shape[1], 3), dtype=np.uint8)
        contours = self.find_and_sort_contours(modified_mask)
        contour_areas, mean_area = mean_area_contours(contours)
        self.crop_uint8 = crop_uint8
        self.modified_mask = modified_mask
        for contour in contours:
            self.process_contour(contour, mean_area, fname, j)




    def preprocess_crop(self, crop):
        
        inverted_mask = adaptive_thresholding_and_invert(crop)
        thresholded_denoised_mask = reduce_noise_and_threshold(inverted_mask)
        output_image_with_huber_curve, output_image_contours_under_huber = fit_curve_with_huber_loss(thresholded_denoised_mask)
        
        self.inverted_mask = inverted_mask
        self.thresholded_denoised_mask = thresholded_denoised_mask
        self.output_image_with_huber_curve = output_image_with_huber_curve
        self.output_image_contours_under_huber = output_image_contours_under_huber
        return output_image_contours_under_huber

    def find_and_sort_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sorted(contours, key=lambda c: np.mean(c[:,:,0]))

     
    def process_contour(self, contour, mean_area, fname, j):
        #_, _, _, rectangle_image = visualize_perfect_rectangle(contour)
        
        if cv2.contourArea(contour) <= mean_area:
            color = (0, 0, 255)   
            #self.CODE_DICT[fname]['code'][j].append(0)
        else:
            color = (0, 255, 0)   
            #self.CODE_DICT[fname]['code'][j].append(1)
        cv2.drawContours(self.final_image, [contour], -1, color, -1)

    def process_image(self,fname, img, save_figures=False, plot_figures=False):
        try:
            crops = [img]
            
            self.figs=[]
            for j, crop in enumerate(crops):
                self.process_crop(crop, "g", j)
                if plot_figures or save_figures:
                    self.visualize_image_processing()
            
            if save_figures:
                self.save_images(fname)
        except Exception as e:
            print(e)
            self.logger.critical(f"Image failed due to {e}")
            #print(f"Image {path} failed due to {e}")
            #if self.full_traceback:
                #tb = traceback.format_exc()
                #print(tb)
            tb = traceback.format_exc()
            self.logger.info(tb)

            

    def visualize_image_processing(self,):
        fig,axs = plt.subplots(3,2,figsize=(30,5))
        axs = axs.ravel()
        axs[0].title.set_text('Raw image')
        axs[0].imshow(self.crop_uint8)

        axs[1].title.set_text('Adapting thresholding and inversion')
        axs[1].imshow(self.inverted_mask)

        axs[2].title.set_text('Denoising')
        axs[2].imshow(self.thresholded_denoised_mask)
        axs[3].title.set_text('Fitted line with huber loss')
        axs[3].imshow(cv2.cvtColor(self.output_image_with_huber_curve, cv2.COLOR_BGR2RGB))
        
        axs[4].title.set_text("Contours under huber")
        axs[4].imshow(cv2.cvtColor(self.output_image_contours_under_huber, cv2.COLOR_BGR2RGB))

        #axs[5].imshow(cv2.cvtColor(contour_and_rectangle_image, cv2.COLOR_BGR2RGB))
        axs[5].imshow(cv2.cvtColor(self.final_image, cv2.COLOR_BGR2RGB))
        #plot_contour_area_histogram(self.contours)
        plt.close('all')
        self.figs.append(fig)


    def save_images(self,fname):
        if not os.path.exists(self.outputs_path):
            os.makedirs(self.outputs_path)
        #text_to_add = f"Message: {msg}\nPrevious message: {results['msg']}"
        #figs[2]= add_text_below_figure(figs[2], text_to_add)
        output_figs_path = os.path.join(self.outputs_path,fname)
        if os.path.exists(output_figs_path):
            self.logger.info(f"fname {fname} already in CODE_DICT. Continuing...")
            #print(f"Skipping saving detailed image. Image {fname} already exists at {self.outputs_path}.")
        else:
            save_figures_as_image(self.figs,output_figs_path)
            self.logger.info(f"Saved detailed figure at {output_figs_path}.")
            #print(f"Saved detailed figure at {output_figs_path}.")
        plt.close('all')


    