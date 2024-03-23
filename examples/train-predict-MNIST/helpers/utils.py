import numpy as np
from halo import Halo
import os, sys, cv2, requests

SCALE = 1 # scale the ui

# scale function
def s(unit):
  return unit * SCALE

def load_data(path):
    """
    Load the data from path to .npz file 
    
    Args: 
        path (list): the path to the .npz file
        
    Returns:
        tuple:
            list: The training data
            list: The labels for this training data
        tuple:
            list: The test data
            list: The labels for this test data
    """
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)
    
def normalize_img(img):
    """
    Normalize  the image 
    
    Args:
        img (list): The image whose values are gonna be normalized
    
    Returns: 
        list: The normalized image
    """
    # Ensure the input array is of data type float32 to support division
    normalized_img = [ row.astype(np.float32) for row in img]
    
    # Divide all elements of the array by 255
    normalized_img = [ row / 255.0 for row in normalized_img]
    
    return normalized_img

def get_accuracy(predictions, labels):
    """
    Compares the predictions and the labels to compute the accuracy 
    
    Args: 
        predictions (list): The predictions made by the model
        labels (list): The true labels
    
    Returns:
        tuple:
            float: The accuracy of the predictions
            integer: The total number of predictions
            integer: The total number of correct predictions
    """
    correct = 0
    n = len(predictions)
    for i in range(n):
        correct += 1 if predictions[i] == labels[i]  else 0
    return (correct / n, n, correct)

def add_module_path_to_sys_path(file):
    # Getting the path to the .so file 
    script_dir = os.path.dirname(os.path.abspath(file))

    so_dir = os.path.join(script_dir, "..", "..", "..", "build")

    # Adding the path to the build dir to the sys.path
    sys.path.append(so_dir)
    
    return so_dir

def file_exists(file_name):
    """
    Check whether a file exists at a specified location

    Args:
        file_name (string): The path and the name of the file to check
    
    Returns:
        boolean: If the file exists (True) or not (False)
    """
    return os.path.exists(file_name)

def get_MNIST_dataset(file_name):
    """
    Fetches the mnist dataset and writes it in the specified .npz file
    
    Args:
        file_name (string): The path and the name to the .npz file to write
    """
    mnist_dataset_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    spinner = Halo(text="Downloading the dataset", spinner="dots")
    spinner.start()
    response = requests.get(mnist_dataset_url)
    spinner.stop()
    with open(file_name, 'wb') as f:
        f.write(response.content)
        

def format_image_grayscale(img, size):
    """
    Resizes the image and returns its resized grayscale value
    
    Args: 
        img (npArray): the image 
        size (tuple):
            height (integer): the desired height
            width (integer): the desired width
    
    Returns:
        npArray: The resized image's grayscale values
    """
    resized_image = cv2.resize(img, size)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Invert grayscale image (for white pixels = 0)
    inverted_gray_image = cv2.bitwise_not(gray_image)
    return inverted_gray_image

def find_highest_indexes_in_matrix(matrix):
    """
    Finds the highest indexes in each row of a numpy matrix
    
    Args:
        matrix (npArray): the matrix
    
    Returns: 
        list: The indexes of the highest values on each row of the matrix
    """
    # Use numpy.argmax to find the index of the maximum value in each row
    highest_indexes = np.argmax(matrix, axis=1)

    return highest_indexes

COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0)
}
