



def countFlies(inputImage):
    '''
    This is the main function that will be called by the main code to count flies given in an image

    The steps are as follows:
        - process image
        - count objects
        - returns an integer (number of flies)
    '''

    processedImage = imagePrepare(inputImage)
    numOfFlies = objCount(processedImage)

    return numOfFlies


def imagePrepare():
    '''
    This function will take an input image as an argument and will perform all the processing before the counting of objects
    It will:
        - Threshold
        - Reduce noise
        - Remove artifacts
        - Identify FOV (maybe, need to think about it)
    '''
    return True

def objCount():
    '''
    This function will count objects in a given image
    It will:
        - segment between objects
        - assign minimum & maximum size
        - return an integer (number of objects)
    '''
    num = 0

    return num