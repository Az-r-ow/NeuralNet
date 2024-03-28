import pygame
import pygame_gui
from helpers.utils import COLORS, s
from helpers.event_handlers import *

# Initialize Pygame
pygame.init()

# Dimensions
margin = s(10)
screen_width = s(640)
screen_height = s(480)
button_width = s(100)
button_height = s(50)
guess_text_width = s(200)
guess_text_height = s(75)
drawing_surface_width = drawing_surface_height = s(300)
drawing_surface_x = (screen_width - drawing_surface_width) // 2
drawing_surface_y = (screen_height - drawing_surface_height) // 2
dropdown_width = s(100)
dropdown_height = s(20)

drawing_color = COLORS["black"]
button_color = COLORS["green"]

DROPDOWN_OPTIONS = [str(num) for num in range(10)]

# Create the display surface
main_window = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Draw a number and make it guess")

drawing_surface = pygame.Surface((drawing_surface_width, drawing_surface_height))
drawing_surface.fill(COLORS["white"])
drawing_surface.set_clip(None)

manager = pygame_gui.UIManager((screen_width, screen_height))
guess_button_rect = pygame.Rect(0, 0, button_width, button_height)
guess_button_rect.bottomright = (-margin, -margin)
guess_button = pygame_gui.elements.UIButton(relative_rect=guess_button_rect, text="Guess it", manager=manager, anchors={'right': 'right', 'bottom': 'bottom'})
clear_button_rect = pygame.Rect((-(margin + (button_width * 2))), -(margin + button_height), button_width, button_height)
clear_button = pygame_gui.elements.UIButton(relative_rect=clear_button_rect, text="Clear", manager=manager, anchors={'right': 'right', 'bottom': 'bottom'})
guess_text_rect = pygame.Rect(0, margin, guess_text_width, guess_text_height)
guess_text = pygame_gui.elements.UITextBox(html_text="", relative_rect=guess_text_rect, manager=manager, anchors={'centerx': 'centerx'})
learn_button_rect = pygame.Rect(margin, margin, button_width, button_height)
learn_button = pygame_gui.elements.UIButton(relative_rect=learn_button_rect, text="Learn It", manager=manager, anchors={'left': 'left', 'top': 'top'})
dropdown_rect = pygame.Rect(0, 0, dropdown_width, dropdown_height)
dropdown_rect.topleft = (margin, margin + button_height)
dropdown = pygame_gui.elements.UIDropDownMenu(DROPDOWN_OPTIONS, DROPDOWN_OPTIONS[0], dropdown_rect, manager=manager, anchors={'left': 'left', 'top': 'top'})

# This dict will be passed to event handlers
ui_elements = {
  "guess_button": guess_button,
  "guess_text": guess_text,
  "drawing_surface": drawing_surface,
  "learn_button": learn_button,
  "dropdown": dropdown,
  'clear_button': clear_button
}

# Fill the main_window with white
main_window.fill(COLORS["black"])
pygame.display.flip()

clock = pygame.time.Clock()
running = True
while running:
  time_delta = clock.tick(60) / 1000.0
  for event in pygame.event.get():
    try: 
      if event.type in EVENT_HANDLER_MAP:
        context = {
          "event": event,
          "main_window": main_window,
          "ui_elements": ui_elements,
          "time_delta": time_delta,
          "drawing_surface_coord": (drawing_surface_x, drawing_surface_y)
        }
        EVENT_HANDLER_MAP[event.type](context)
    except KeyError as e:
      print(f"A key error happened: {event.type}")
      print(e)
    manager.process_events(event)
  
  main_window.blit(drawing_surface, (drawing_surface_x, drawing_surface_y))
  manager.update(time_delta)
  manager.draw_ui(main_window)
  pygame.display.flip()
