import numpy as np
import pandas as pd
import cv2


def pick_image(row_data, steering_correction):
    '''
    Picks an image from amongst the center, left or right
    side.
    
    Parameters:
        - row_data: CSV row data
        - steering_correction: Correction of the steering angle
    
    Output:
        - img_path: Path of the picked image
        - steering: Corrected steering value
    '''
    # Randomly pick one option
    toss = np.random.randint(3)
    
    # Values to return
    img_path = ''
    steering = 0.0
    
    # Center image
    if toss == 0:
        img_path = row_data.iloc[0]
        steering = row_data.iloc[3]
    
    # Left image
    elif toss==1:
        img_path = row_data.iloc[1]
        steering = row_data.iloc[3] + steering_correction
    
    # Right image
    elif toss==2:
        img_path = row_data.iloc[2]
        steering = row_data.iloc[3] - steering_correction
    
    return img_path, steering


def load_image(img_path):
    '''
    Loads an image by reading from a file path
    
    Parameters:
        - img_path: path of the image to load
    
    Output:
        - RGB image
    '''
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image


def translate_image(image, steering_angle, range_x = 100, range_y = 10):
    """
    Randomly shift the image vertically and horizontally.
    
    Parameters:
        - image: Image to be shifted
        - steering_angle: Steering angle
        - range_x: Range of transformation for the x axis
        - range_y: Range of transformation for the y axis
    
    Output:
        - image: Warped image
        - steering_angle : Compensated steering angle
    """
    # Random translation values
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    # Compensate steering angle
    steering_angle += trans_x * 0.002
    # Transformation matrix
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    # Warp it
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def increase_brightness(img):
    '''
    Randomly adds brightness to an image.
    
    Parameters:
        - img: Image to treat

    Output:
        - image: Brighter image
    '''
    # Transform to HSV
    imgage = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    # Pick a random value for brightness
    random_bright = .25 + np.random.uniform()
    imgage[:,:,2] = imgage[:,:,2] * random_bright
    # Transform back to RGB
    imgage = cv2.cvtColor(imgage, cv2.COLOR_HSV2RGB)
    return imgage


def random_shadow(image):
    """
    Generates a random shadow in an image.
    
    Parameters:
        - img: Image to treat

    Output:
        - image with shadow
    """
    height, width = image.shape[:2]
    
    # (x1, y1) and (x2, y2) forms a line. xm, ym gives all the locations of the image
    x1, y1 = width * np.random.rand(), 0
    x2, y2 = width * np.random.rand(), height
    xm, ym = np.mgrid[0:height, 0:width]

    # Mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # Choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low = 0.2, high = 0.5)

    # Adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def crop_image(image, vertical):
    '''
    Crop image vertically
    
    Parameters:
        - image: Image to crop
        - vertical: Top and bottom points to crop
    
    Output:
        - cropped image
    '''
    cropped_image = image[vertical[0]:vertical[1], :, :]
    return cropped_image


def resize_image(image, size):
    '''
    Resize an image to the given size
    
    Parameters:
        - image: Image to be resized
        - size: New size of the image
    
    Output:
        - resized image
    '''
    return cv2.resize(image, size)

    
def convert_to_HSV(image):
    '''
    Converts an image from RGB to HSV colorspace
    
    Parameters:
        - img: RGB image to be converted
    
    Output:
        - HSV image
    '''
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return image


def preprocess_image(image, vertical_crop, size):
    '''
    Steps for preprocessing an image before feeding it to 
    the network. It crops, resizes and converts the image
    to HSV colorspace.
    
    Parameters:
        - image: Image to preprocess
        - vertical_crop: Top and bottom points to crop
        - size: New size of the image
    
    Output:
        - preprocessed image
    '''
    image = crop_image(image, vertical_crop)
    image = resize_image(image, size)
    image = convert_to_HSV(image)
    image = np.array(image)
    return image



def perturb_image(data_path, row_data, steering_correction, vertical_crop, size):
    '''
    Perturbs an image by translating, adding random shadow, increasing
    brightness and preprocessing (crop , resize and conversion to HSV space) it.
    
    Parameters:
        - data_path: Path of the data (root)
        - row_data: CSV row data
        - steering_correction: Correction of the steering angle
        - vertical_crop: Top and bottom points to crop
        - size: New size of the image
        
    Output:
        - image: Perturbed image
        - steering: Steering value of the image
    ''' 
    # Pick an image
    img_path, steering = pick_image(row_data, steering_correction)
    image = load_image(data_path + img_path.strip())

    # Translate image
    image, steering = translate_image(image, steering)
    
    # Increase brightness
    image = increase_brightness(image)

    # Flip image. This is done to reduce the bias for 
    # turning left that is present in the training data
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        # Flip the image and reverse the steering angle
        steering = -1 * steering
        image = cv2.flip(image, 1)

    # Random shadow
    image = random_shadow(image)

    # Preprocess image
    image = preprocess_image(image, vertical_crop, size)
   
    return image, steering
