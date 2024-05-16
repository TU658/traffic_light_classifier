import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

import test_functions

import time

IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

# image_num = 750
# The first image in IMAGE_LIST is displayed below (without information about shape or label)
# selected_image = IMAGE_LIST[image_num][0]
# label_image = IMAGE_LIST[image_num][1]
# print('Shape of the image : \t',selected_image.shape)
# print('Label of the image : \t',label_image)
# plt.imshow(selected_image)

new_width = 35
new_height = 35
def standardize_input(image):
    
    ##Resize image and pre-process so that all "standard" images are the same size  
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (new_width, new_height)) 
    
    return standard_im

standarded_im = list()

for im in range(len(IMAGE_LIST)):
    standarded_im.append(standardize_input(IMAGE_LIST[im][0]))

# test:
# print(standarded_im[500].shape)
# plt.imshow(standarded_im[500])

## Given a label - "red", "green", or "yellow" - return a one-hot encoded label

# Examples: 
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]

def one_hot_encode(label):
    
    ##Create a one-hot encoded label that works for all classes of traffic lights
    one_hot_encoded = [] 
    if(label == 'red'):
        one_hot_encoded = [1, 0, 0]
    elif(label == 'green'):
        one_hot_encoded = [0, 0, 1]
    else:
        one_hot_encoded = [0, 1, 0]
    return one_hot_encoded

# test:
# label = 'yellow'
# print('label of image: ', one_hot_encode(label))

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

# print('image standarded')
# print(STANDARDIZED_LIST[750][1])
# print(STANDARDIZED_LIST[750][0].shape)
# plt.subplot(1, 2, 1)
# plt.title('Image standarded')
# plt.imshow(STANDARDIZED_LIST[750][0])

# print('image nonstandard')
# print(IMAGE_LIST[750][1])
# print(IMAGE_LIST[750][0].shape)
# plt.subplot(1, 2, 2)
# plt.title('Image nonstandard')
# plt.imshow(IMAGE_LIST[750][0])

def create_feature(rgb_image):
    image_copy = np.copy(rgb_image)
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([131, 166, 201])
    
    lower_white = np.array([122, 122, 122])
    upper_white = np.array([230, 200, 207])
    
    mask_white = cv2.inRange(image_copy, lower_white, upper_white)
    mask_black = cv2.inRange(image_copy, lower_black, upper_black)
    
    image_copy[mask_black != 0] = [0, 0, 0]
    image_copy[mask_white != 0] = [0, 0, 0]
    
    cropped_image = image_copy[7:31 , 10:27]
    ##Convert image to HSV color space
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
    value = hsv[:,:,2]
    ##Create and return a feature value and/or vector
    feature = []
    feature = np.sum(value, axis = 1)
    brightness_row = np.sum(value, axis = 1)
    column = np.arange(len(value))
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax3.barh(column, brightness_row)
    ax3.invert_yaxis()

    ax2.imshow(cropped_image)

    ax1.imshow(rgb_image)


    return feature

#test:

# print(create_feature(STANDARDIZED_LIST[750][0]))

# image_test = 840
# # print('avg_brightness of image', create_feature(STANDARDIZED_LIST[image_test][0]))
# # print(STANDARDIZED_LIST[image_test][1])
# # print(STANDARDIZED_LIST[image_test][0].shape)
# # plt.imshow(STANDARDIZED_LIST[image_test][0])

# selected_image = STANDARDIZED_LIST[image_test][0]
# # cropped_image = selected_image[6:33 , 8:29]
# # print(cropped_image.shape)
# # plt.imshow(cropped_image)
# # create_feature(selected_image)

# lower_black = np.array([0, 0 ,0])
# upper_black = np.array([194, 194, 194])

# mask = cv2.inRange(selected_image, lower_black, upper_black)

# selected_image[mask != 0] = [0, 0, 0]
# # masked_image = cv2.bitwise_and(selected_image, selected_image, mask = mask)

# # f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 20))
# # ax1.imshow(masked_image)
# # ax1.set_title('masked_image')
# # ax2.imshow(selected_image)
# # ax2.set_title('original image')
# cropped_image = selected_image[6:32 , 10:27]
# hsv = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
# width, height, channel = cropped_image.shape
# print(width, height)
# # average_brightness = np.sum(hsv[:, :, 2])/(width * height)

# hsv = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
# value = hsv[:,:,2]
# figure = np.array(value)
# print(figure)
# brightness_row = np.sum(value, axis = 1)
# for i in range(len(brightness_row)):
#     print(brightness_row[i])
# column = np.arange(len(value))
# f, (ax1, ax2) = plt.subplots(1, 2)
# ax2.barh(column, brightness_row)
# ax2.invert_yaxis()

# ax1.imshow(cropped_image)

# (Optional) Add more image analysis and create more features
def rgb_histograms(rgb_image):

    red_lower = np.array([171, 0, 24])
    red_upper = np.array([235, 148, 160])

    
#     yellow_lower = np.array([0, 0, 0])
#     yellow_upper = np.array([0, 0, 0])
    
    green_lower = np.array([0, 128, 107])
    green_upper = np.array([99, 194, 164])
    
    image_rgb = rgb_image[2:34 , 13:32]
    
    mask_red = cv2.inRange(image_rgb, red_lower, red_upper)
    mask_green = cv2.inRange(image_rgb, green_lower, green_upper)
#     mask_yellow = cv2.inRange(image_rgb, yellow_lower, yellow_upper)
    
    red_hist = np.histogram(image_rgb[mask_red > 0])
    green_hist = np.histogram(image_rgb[mask_green > 0])
    
    
#     red_hist = np.histogram(image_rgb[mask_red > 0], bins=256, range=(0, 256))
#     green_hist = np.histogram(image_rgb[mask_green > 0], bins=256, range=(0, 256))
#     yellow_hist = np.histogram(image_rgb[mask_yellow > 0], bins=256, range=(0, 256))

    sum_red_hist = np.sum(red_hist[0])
#     sum_yellow_hist = np.sum(yellow_hist[0])
    sum_green_hist = np.sum(green_hist[0])  

    plt.figure(figsize=(10, 5))
  
    plt.subplot(1, 5, 1)
    plt.plot(red_hist[1][:-1], red_hist[0], color='red')
    plt.title('Red Color')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

#     plt.subplot(1, 6, 2)
#     plt.plot(yellow_hist[1][:-1], yellow_hist[0], color='yellow')
#     plt.title('yellow Color')
#     plt.xlabel('Intensity')
#     plt.ylabel('Frequency')
    
    plt.subplot(1, 5, 2)
    plt.plot(green_hist[1][:-1], green_hist[0], color='green')
    plt.title('Green Color')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 5, 3)
    plt.imshow(mask_red, cmap = 'gray')
    
#     plt.subplot(1, 6, 5)
#     plt.imshow(mask_yellow, cmap = 'gray')
    
    plt.subplot(1, 5, 4)
    plt.imshow(mask_green, cmap = 'gray')
    
    plt.subplot(1, 5, 5)
    plt.imshow(image_rgb)
    
    plt.show()
    
    print('red: ', sum_red_hist)
#     print('yellow: ', sum_yellow_hist)
    print('green: ', sum_green_hist)
    
    return sum_red_hist, sum_green_hist


# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):
    
    ##Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    predicted_label = []
    
    feature = create_feature(rgb_image)
    value_red, value_green = rgb_histograms(rgb_image)
    
    area_red = np.sum(feature[:8])
    area_yellow = np.sum(feature[8:13])
    area_green = np.sum(feature[19:])
    
    if(area_red > area_yellow and area_red > area_green):
        if(value_red >= value_green):
            predicted_label = [1, 0, 0]
        else:
            predicted_label = [0, 1, 0]
    elif(area_green > area_yellow and area_green > area_red):
        if(value_red > value_green):
            predicted_label = [0, 1, 0] 
        else:
            predicted_label = [0, 0, 1]
    elif(area_green == area_yellow and area_green == area_red):
        if(value_red > value_green):
            predicted_label = [1, 0, 0]
        elif(value_red < value_green):
            predicted_label = [0, 0, 1]
        else:
            predicted_label = [0, 1, 0]
    else:
        predicted_label = [0, 1, 0]

    print('area_red: ', area_red, '\t area_yellow: ', area_yellow, '\t area_green: ', area_green)

    print('predicted_label : ', "RED" if predicted_label == [1, 0, 0] else ("GREEN" if predicted_label == [0, 0, 1] else "YELLOW" ))
    
    return predicted_label   

##test
# print(estimate_label(STANDARDIZED_LIST[740][0]))


# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
#             print(predicted_label, true_label, '\n')
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

tests = test_functions.Tests()
if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")

