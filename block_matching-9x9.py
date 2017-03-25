import cv2
import numpy as np

#read the input images of the 2 views as gray-scale images
img_lft = cv2.imread('view1.png', 0)
img_rght = cv2.imread('view5.png', 0)
#padding the images
img_lft_padded = cv2.copyMakeBorder(img_lft, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0);
img_rght_padded = cv2.copyMakeBorder(img_rght, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0);

#calculating disparity map for left image
width, height = img_lft_padded.shape
disp_map_lft = np.zeros((width, height))
for i in range(5, width-4):
    for j in range(5, height-4):
        current_block_lft = img_lft_padded[i-4:i+5, j-4:j+5]
        match_lft = 0
        min_ssdlft = np.iinfo.max  # initialize left image min ssd as maximum value of int in numpy
        for idx in range(j-75, j):
            # to avoid over-flow error while slicing right view image into blocks
            if idx < 5:
                idx = 5
            current_ssd = np.sum(np.square(np.subtract(current_block_lft, img_rght_padded[i-4:i+5, idx-4:idx+5])))

            if current_ssd < min_ssdlft:
                min_ssdlft = current_ssd
                match_lft = idx

        disp_map_lft[i][j] = j - match_lft

#calculating disparity map for the right image
width, height = img_rght_padded.shape
disp_map_rght = np.zeros((width, height))
for i in range(5, width-4):
    for j in range(5, height-4):
        current_block_rght = img_rght_padded[i-4:i+5, j-4:j+5]
        match_rght = 0
        min_ssdrght = np.iinfo.max  # initialize right image min ssd as maximum value of int in numpy
        for idx in range(j + 75, j, -1):
            # to avoid over-flow error while slicing left view image into blocks
            if idx >= height-5:
                idx = height-6
            current_ssd = np.sum(np.square(np.subtract(current_block_rght, img_lft_padded[i-4:i+5, idx-4:idx+5])))

            if current_ssd < min_ssdrght:
                min_ssdrght = current_ssd
                match_rght = idx

        disp_map_rght[i][j] = match_rght - j

#resize the disparity image to original image width, height
disp_map_lft = disp_map_lft[4:width-4, 4:height-4]
disp_map_rght = disp_map_rght[4:width-4, 4:height-4]

#MSE calculation for the left and right disparity images w.r.t ground truth disparity
# reference: https://www.mathworks.com/matlabcentral/fileexchange/37854-mse-reference-image--target-image-/content/MSE.m
#read the input images of the 2 ground truth disparity as gray-scale images
gdt_disp_lft = cv2.imread('disp1.png', 0)
gdt_disp_rght = cv2.imread('disp5.png', 0)

mse_left = np.sum(np.square(np.subtract(gdt_disp_lft, disp_map_lft)))
mse_right = np.sum(np.square(np.subtract(gdt_disp_rght, disp_map_rght)));

mse_left = mse_left/(width*height)
mse_right = mse_right/(width*height)

print 'MSE for disparity left 9x9 is: ', mse_left
print 'MSE for disparity right 9x9 is: ', mse_right

cv2.namedWindow('Left Disparity 9x9', cv2.WINDOW_NORMAL)
cv2.imshow("Left Disparity 9x9", disp_map_lft/disp_map_lft.max())
cv2.namedWindow('Right Disparity 9x9', cv2.WINDOW_NORMAL)
cv2.imshow("Right Disparity 9x9", disp_map_rght/disp_map_rght.max())
cv2.waitKey(0)
cv2.destroyAllWindows()
