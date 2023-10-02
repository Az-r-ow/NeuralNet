"""
  In this file, we : 
  - Load the previously trained model
"""
import sys, os , requests, random
from utils import * 

# Adding the module path to the sys path 
so_dir = add_module_path_to_sys_path(__file__)

import NeuralNetPy as NNP

MY_SAMPLES_FOLDER = "./dataset/my_samples"

network = NNP.Network()

# Loading the model from the file into the network created
NNP.Model.load_from_file("model.bin", network)

inputs = list()

sample_files = os.listdir(MY_SAMPLES_FOLDER)

for filename in sample_files:
  file_path = os.path.join(MY_SAMPLES_FOLDER, filename)
  image = cv2.imread(file_path)
  image_greyscale = format_image_greyscale(image, (28, 28))
  normalized_image = normalize_img(image_greyscale.flatten())
  inputs.append(normalized_image)

predictions = network.predict(inputs)

for i in range(len(predictions)):
  print(f"Prediction : {predictions[i]} - Actual value : {sample_files[i]}")

# Remove sys.path modification
sys.path.remove(so_dir)