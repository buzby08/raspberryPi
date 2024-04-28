import cv2
import matplotlib.pyplot as plt
import cvlib
from cvlib.object_detection import draw_bbox

def image_Object_Detection(image_file):
	# reading the image
    img = cv2.imread(image_file)
    # detect common objects
    bbox, label, conf = cvlib.detect_common_objects(img)
    # draw bbox around the objects
    output_image = draw_bbox(img, bbox, label, conf)
    # show and plot the image
    plt.imshow(output_image)
    plt.show()
	
	# print the number of different labels found in the images
    for item in set(label):
        print(f'Number of {item} in the image is {str(label.count(item))}')
        
if __name__=="__main__":
	# list image paths
    image_list = ['cats_dogs.png','people.jpeg']
    # call the above function for each image
    for img in image_list:
        image_Object_Detection(img)
