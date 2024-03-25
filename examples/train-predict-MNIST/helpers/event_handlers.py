import pygame
import pygame_gui
import sys
import numpy
from helpers.utils import *

so_dir = add_module_path_to_sys_path(__file__)

import NeuralNetPy as NNP

network = NNP.models.Network()

NNP.models.Model.load_from_file("model.bin", network)

drawing = False # Track drawing
erasing = False # Track erasing
drawing_color = COLORS["black"]
erasing_color = COLORS["white"]


def handle_ui_button_pressed(context):
  if context["event"].ui_element == context["ui_elements"]["guess_button"]:
    drawing_image = pygame.surfarray.pixels3d(context["ui_elements"]["drawing_surface"])
    grayscale_image = format_image_grayscale(drawing_image, (28, 28))
    normalized_image = normalize_img(numpy.transpose(grayscale_image))
    prediction = find_highest_indexes_in_matrix(network.predict([normalized_image]))
    context["ui_elements"]["guess_text"].append_html_text(f"I'm guessing : {prediction[0]}<br>")
  return
  
def handle_mouse_button_down(context):
  global drawing, erasing 
  if context["event"].button == 1: # Left click
    drawing = True
  if context["event"].button == 3: # Right click to erase
    erasing = True

def handle_mouse_button_up(context):
  global drawing, erasing
  if context["event"].button == 1: # Left click 
    drawing = False
  if context["event"].button == 3: # Right click
    erasing = False

def handle_mouse_motion(context):
  # Draw or erase depending on state
  drawing_surface_x, drawing_surface_y = context["drawing_surface_coord"]
  drawing_surface_rect = context["ui_elements"]["drawing_surface"].get_rect(topleft=(drawing_surface_x, drawing_surface_y))
  
  if drawing_surface_rect.collidepoint(context["event"].pos):
    mouse_x, mouse_y = context["event"].pos
    pos = (mouse_x - drawing_surface_x, mouse_y - drawing_surface_y)
    if drawing:
      pygame.draw.circle(context["ui_elements"]["drawing_surface"], drawing_color, pos, s(5))
    if erasing:
      pygame.draw.circle(context["ui_elements"]["drawing_surface"], erasing_color, pos, s(10))

def handle_quit(context):
  # Remove sys.path modification
  sys.path.remove(so_dir)
  pygame.quit()
  sys.exit()
  
EVENT_HANDLER_MAP = {
  pygame_gui.UI_BUTTON_PRESSED: handle_ui_button_pressed,
  pygame.MOUSEBUTTONDOWN: handle_mouse_button_down,
  pygame.MOUSEBUTTONUP: handle_mouse_button_up,
  pygame.MOUSEMOTION: handle_mouse_motion,
  pygame.QUIT: handle_quit
}
