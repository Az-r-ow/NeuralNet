import numpy as np

def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)
    
def normalize_img(img):
    # Ensure the input array is of data type float32 to support division
    normalized_img = img.astype(np.float32)
    
    # Divide all elements of the array by 255
    normalized_img /= 255.0
    
    return normalized_img

