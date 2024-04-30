import cv2
import numpy as np
import math

def hdl(generated, true):

   true_x_shifted_right = true[1:,:,:]
   true_x_shifted_left = true[:-1,:,:]
   true_x_gradient = np.abs(true_x_shifted_right - true_x_shifted_left)

   generated_x_shifted_right = generated[1:,:,:]
   generated_x_shifted_left = generated[:-1,:,:]
   generated_x_gradient = np.abs(generated_x_shifted_right - generated_x_shifted_left)
   
   loss_x_gradient = np.linalg.norm(true_x_gradient - generated_x_gradient)

   true_y_shifted_right = true[:,1:,:]
   true_y_shifted_left = true[:,:-1,:]
   true_y_gradient = np.abs(true_y_shifted_right - true_y_shifted_left)

   generated_y_shifted_right = generated[:,1:,:]
   generated_y_shifted_left = generated[:,:-1,:]
   generated_y_gradient = np.abs(generated_y_shifted_right - generated_y_shifted_left)
    
   loss_y_gradient = np.linalg.norm(true_y_gradient - generated_y_gradient)

   loss = loss_x_gradient + loss_y_gradient

   return loss

def psnr(img1, img2):
   mse = np.mean( (img1 - img2) ** 2 )
   return 10 * math.log10(255.0**2/mse)

for i in range(1,5):

   Uimage = cv2.imread('result_images/Uimage_crop'+str(i)+'_og.png')
   Rchannel   = cv2.imread('result_images/Rchannel_crop'  +str(i)+'_og.png')
   Gchannel   = cv2.imread('result_images/Gchannel_crop'  +str(i)+'_og.png')
   Bchannel  = cv2.imread('result_images/Bchannel_crop' +str(i)+'_og.png')

   Uimage = cv2.calcHist([Uimage], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
   Rchannel_hist   = cv2.calcHist([Rchannel],   [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
   Gchannel_hist   = cv2.calcHist([Gchannel],   [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
   Bchannel_hist  = cv2.calcHist([Bchannel],  [0,1,2], None, [8,8,8], [0,256,0,256,0,256])

   a = cv2.compareHist(Uimage_hist, Rchannel_hist,  cv2.HISTCMP_underwater)
   b = cv2.compareHist(Uimage_hist, Gchannel_hist,  cv2.HISTCMP_underwater)
   c = cv2.compareHist(Uimage_hist, Bchannel_hist, cv2.HISTCMP_underwater)

   Uimage = Uimage/255.0
   Rchannel  = Rchannel/255.0
   Gchannel   = Gchannel/255.0
   Bchannel  = Bchannel/255.0

   '''
   print '+--------+'
   print '|  crop'+str(i)+' |'
   print '+--------+'
   print 
   print 'hdl Uimage-Rchannel    :',gdl(Uimage, Rchannel)
   print 'hdl Uimage-Gchannel    :',gdl(Uimage, Gchannel)
   print 'hdl Uimage-Bchannel   :',gdl(Uimage, Bchannel)
   
   print
   '''
   print '& ' + str(round(np.mean(flickr), 2)) + ' $\pm$ ' + str(round(np.std(flickr), 2))  ' \\\ \hline'

