import cv2 as cv
import matplotlib.pyplot as plt 
import numpy as np

test_files = ['images/redscale.png']

edge_kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
# red_kernel = np.array([2,-1,-1])
# green_kernel = np.array([-1,2,-1])
# blue_kernel = np.array([-1,-1,2]) #RGB bad idea aye lmao

red_kernel = np.array([1,0,0])
green_kernel = np.array([0,1,0])
blue_kernel = np.array([0,0,1])
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
    """
    Get a conv matrix using edge detection kernels
    """
    return


def get_histogram(image):
    """
    Extract the histograms related to the input image
    input:
        image: image in default BGR format
    output:

    """
    channels = image.shape[2]
    hists = list()

    bins = [256]
    rng = [0,256]
    # Convert image and get histograms
    rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    r_hist = cv.calcHist([rgb_img], [0], None, bins, rng, accumulate=False)
    g_hist = cv.calcHist([rgb_img], [1], None, bins, rng, accumulate=False)
    b_hist = cv.calcHist([rgb_img], [2], None, bins, rng, accumulate=False)

    # Plot histograms
    hstFig,hstAxs=plt.subplots(1,1)
    hstFig.suptitle("Histograms")
    # hstAxs[0].plot(r_hist, color='r')
    # hstAxs[1].plot(g_hist, color='g')
    # hstAxs[2].plot(b_hist, color='b')
    hstAxs.plot(r_hist, color='r')
    hstAxs.plot(g_hist, color='g')
    hstAxs.plot(b_hist, color='b')

    return

def color_pop(image, threshold):
    """
    Takes a signle image as input and shows a grayscale of each channel
    input:
        image: BGR image
        threshold: unused
    output:

    """
    rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_shape = rgb_img.shape
    channels = image_shape[2]
    
    pop_images = list()

    # Check image dimensions
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
            #break

        # Normalize (to 255)
        for img in pop_images:
            coef = (255/img.max())
            for i in range(image_shape[0]):
                for j in range(image_shape[1]):
                    img[i][j] = round((img[i][j]*coef)[0])    
            #break
    else:
        pass
    
    # Display grayscales for each channel
    cmaps = ['Reds', 'Greens', 'Blues']
    if channels != 1:
        for m in range(len(cmaps)):
            popFig,popAxs=plt.subplots(1,2)
            popFig.suptitle("Grayscale " + cmaps[m] + " channel")
            
            popAxs[0].imshow(pop_images[m], cmap='gray')
            popAxs[1].imshow(pop_images[m], cmap=cmaps[m])
    else:
        pass
    
    return

if __name__ == "__main__":
    main()