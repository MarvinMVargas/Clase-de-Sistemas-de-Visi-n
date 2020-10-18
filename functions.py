import cv2 as cv
import matplotlib.pyplot as plt 
import numpy as np

test_files = ['dice2.jpg']
edge_kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])

red_kernel = np.array([2,-1,-1])
green_kernel = np.array([-1,2,-1])
blue_kernel = np.array([-1,-1,2]) #RGB

rgb_kernels = [red_kernel,green_kernel,blue_kernel]


def main():
    imList = import_images(test_files)
    fig,axs=plt.subplots(1,len(imList))
    fig.suptitle("Imported images")
    
    for img in imList:
        get_histogram(img)
        color_pop(img,0)

    # Plot in graphs
    if len(imList) == 1:
        axs.imshow(cv.cvtColor(imList[0], cv.COLOR_BGR2RGB))
    else:
        for row in range(axs.numRows):
            for col in range(axs.numCols):
                axs[row][col].imshow(cv.cvtColor(imList[img], cv.COLOR_BGR2RGB))

    plt.show()

def import_images(paths):
    """
    Import images from input paths
    inputs:
        paths: list containing paths to images
    output:
        imgs: list containing cv2 objects
    """
    imgs = list()
    for path in paths:
        imgs.append(cv.resize(cv.imread(path),(400,300)))
    return imgs

def edge_recognition(images, kenel, padding=False):
    return


def get_histogram(image):
    channels = image.shape[2]
    hists = list()

    bins = [256]
    rng = [0,256]
    rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    r_hist = cv.calcHist([rgb_img], [0], None, bins, rng, accumulate=False)
    g_hist = cv.calcHist([rgb_img], [1], None, bins, rng, accumulate=False)
    b_hist = cv.calcHist([rgb_img], [2], None, bins, rng, accumulate=False)
    hstFig,hstAxs=plt.subplots(1,1)
    hstFig.suptitle("Histograms")
    # hstAxs[0].plot(r_hist, color='r')
    # hstAxs[1].plot(g_hist, color='g')
    # hstAxs[2].plot(b_hist, color='b')
    hstAxs.plot(r_hist, color='r')
    hstAxs.plot(g_hist, color='g')
    hstAxs.plot(b_hist, color='b')
    #plt.show()

    #chan1 = image.


    return

def color_pop(image, threshold):
    rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_shape = rgb_img.shape
    channels = image_shape[2]
    
    pop_images = list()

    if channels != 1:
        for kernel in rgb_kernels:
            pop_img = np.zeros((300,400,1), np.uint8)
            for i in range(image_shape[0]):
                for j in range(image_shape[1]):
                    # Perform convolution
                    conv = rgb_img[i][j]*kernel
                    res = conv.sum()
                    pop_img[i][j] = res
            pop_images.append(pop_img)
            break
        # Normalize (to 255)
        for img in pop_images:
            #coef = (255/img.max())
            coef = (1/img.max())
            for i in range(image_shape[0]):
                for j in range(image_shape[1]):
                    img[i][j] = round((img[i][j]*coef)[0])
                    
            break

    popFig,popAxs=plt.subplots(1,2)
    popFig.suptitle("Pop Image")
    
    popAxs[0].imshow(pop_img, cmap='gray')
    popAxs[1].imshow(pop_img, cmap='Reds')
    #plt.show()  

    return


if __name__ == "__main__":
    main()