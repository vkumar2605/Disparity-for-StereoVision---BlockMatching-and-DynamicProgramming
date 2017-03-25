import numpy as np
import cv2
from matplotlib import pyplot as plt

# Disparity Calculation as per the below paper
# Cox, Ingemar J., et al. "A maximum likelihood stereo algorithm." Computer vision and
# image understanding 63.3 (1996): 542-567

# MSE calculation function
def mse(leftDisp,rightDisp,occ):
    leftGroundTruthImg = cv2.imread("disp1.png",0)
    leftGroundTruthImg = np.asarray(leftGroundTruthImg, dtype=np.uint8) 
    rightGroundTruthImg = cv2.imread("disp5.png",0)
    rightGroundTruthImg = np.asarray(rightGroundTruthImg, dtype=np.uint8)
    
    rows = leftGroundTruthImg.shape[0]
    cols = leftGroundTruthImg.shape[1]
    
    mseValueLeft=0 
    mseValueRight=0 
    
    for i in range (0,rows):
        for j in range(0,cols):
            if(leftDisp[i][j]!=0):
               mseValueLeft += np.square(rightGroundTruthImg[i][j]-leftDisp[i][j])
            if(rightDisp[i][j]!=0):
               mseValueRight += np.square(rightGroundTruthImg[i][j]-rightDisp[i][j])
    
    
    mseValueLeft = mseValueLeft/(rows*cols)
    mseValueRight = mseValueRight/(rows*cols)
    
    print "Occ Value : "+ str(occ)
    print "MSE_Left Value : "+str(mseValueLeft)
    print "MSE_Right Value : "+str(mseValueRight)
    
# Stereo Matching function as per the given paper
def stereoMatching(leftImg,rightImg):
    rows = leftImg.shape[0]
    cols = leftImg.shape[1]
    
    # Matrices to store disparities : left and right
    leftDisp=np.zeros((rows,cols))
    rightDisp=np.zeros((rows,cols))
     
    occ = 7
    
    # Pick a row in the image to be matched
    for c in range (0,rows):
        # Cost matrix 
        colMat=np.zeros((cols,cols))
        
        # Disparity path matrix
        dispMat=np.zeros((cols,cols))
        
        # Initialize the cost matrix 
        for i in range(0,cols):
            colMat[i][0] = i*occ
            colMat[0][i] = i*occ
        
        # Iterate the row in both the images to find the path using dynamic programming
        # Progamme is similar to LCS(Longest common subsequence)
        
        for k in range (0,cols):
            for j in range(0,cols):        
                if(leftImg[c][k]>rightImg[c][j]):
                    match_cost=leftImg[c][k]-rightImg[c][j]
                else:
                    match_cost=rightImg[c][j]-leftImg[c][k]
                
                # Finding minimum cost    
                min1=colMat[k-1][j-1]+match_cost
                min2=colMat[k-1][j]+occ
                min3=colMat[k][j-1]+occ
                
                colMat[k][j]=cmin=min(min1,min2,min3)
                
                # Marking the path 
                if(min1==cmin):
                    dispMat[k][j]=1
                if(min2==cmin):
                    dispMat[k][j]=2
                if(min3==cmin):
                    dispMat[k][j]=3
        
        # Iterate the matched path and update the disparity value
        i=cols-1
        j=cols-1
        
        while (i!=0) and  (j!=0):
            if(dispMat[i][j]==1):
                leftDisp[c][i]=np.absolute(i-j)
                rightDisp[c][j]=np.absolute(j-i)
                i=i-1
                j=j-1
            elif(dispMat[i][j]==2):
                leftDisp[c][i]=0
                i=i-1
            elif(dispMat[i][j]==3):
                rightDisp[c][j]=0
                j=j-1
                
    # Uncomment the following 2 lines code to write the left and right disparity images
    
    #cv2.imwrite("Left_Disparity_"+str(occ)+".png",leftDisp)
    #cv2.imwrite("Right_Disparity_"+str(occ)+".png",rightDisp)
    
    # Calculate MSE with respect to the ground truth values
    mse(leftDisp,rightDisp,occ)
    
    # Uncomment the following code to display the left and right disparity images
        
    #plt.subplot(111),plt.imshow(displft, cmap = 'gray')
    #plt.title('Left Disparity'), plt.xticks(), plt.yticks()
    #plt.subplot(111),plt.imshow(displft, cmap = 'gray')
    #plt.title('Right Disparity'), plt.xticks(), plt.yticks()
    #plt.show()
        
        
def main():
    
    # Read images.
    leftImg = cv2.imread("view1.png",0)
    leftImg = np.asarray(leftImg, dtype=np.uint8) 
    rightImg = cv2.imread("view5.png",0)
    rightImg = np.asarray(rightImg, dtype=np.uint8)
    
    # Call disparity matching algorithm
    stereoMatching(leftImg,rightImg)
    
main()