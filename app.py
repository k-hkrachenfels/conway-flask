from flask import Flask
from datetime import datetime
import re
import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pylab as plt
import matplotlib.animation
from io import BytesIO
import binascii


app = Flask(__name__)

# small world
A = np.array([
  [0,0,0,0,0,0,0,0],
  [0,0,0,1,0,0,0,0],
  [0,0,0,0,1,0,0,0],
  [0,0,1,1,1,0,0,0],
  [0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0]])

# bigger world
B = np.array([
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])  

# kernel
K = np.asarray([
  [1,1,1],
  [1,0,1],
  [1,1,1]])

def figure_asset(K, growth):
  """Subgraphs showing Kernel, Growth and cross-section
  """
  K_sum = np.sum(K)
  K_size = K.shape[0]
  K_mid = K_size // 2

  matplotlib.use('agg')
  fig, ax = plt.subplots(1, 3, figsize=(14, 2), gridspec_kw={'width_ratios': [1,1,2]})

  ax[0].imshow(K, cmap="jet", interpolation="nearest", vmin=0)
  ax[0].title.set_text('kernel K')

  ax[1].bar(range(K_size), K[K_mid,:], width=1)
  ax[1].title.set_text('Kernel K cross-section')
  ax[1].set_xlim([K_mid - 1 - 3, K_mid + 1 + 3])

  x = np.arange(K_sum + 1)
  ax[2].step(x, growth(x))
  ax[2].axhline(y=0, color='grey', linestyle='dotted')
  ax[2].title.set_text('growth G')
  return fig

def figure_world(A):
  global img
  fig = plt.figure(figsize=(15,15))
  img = plt.imshow(A, cmap="jet", interpolation="nearest", vmin=0)
  plt.title('World A')
  plt.close()
  return fig

def growth(U):
  # U is a 2-dim matrix that in the case of the 'ring'-kernel the count of neighbors with a 1
  # the formula returns
  #   1 when exactly 3 neihbours exist
  #   0 when exactly 2 neighbors exist
  #   -1 otherwise
  # Because this is used as growth to the original value this means
  #   empty fields get filled, when exactly 3 neighbors exist
  #   filled fields survive when 2 or 3 neihbours exist
  #   all other fields die
  return 0 + (U==3) - ((U<2)|(U>3))

def evolve(i):
  global A
  # U is filled with the sum values of the neighbours where the kernel is applied
  # for the ring kernel and the neighbour values either 0 or 1 this means a count 
  # of neighbours 
  U = scipy.signal.convolve2d(A, K, mode='same', boundary='wrap')
  # add the growth value, see description of growth and clip to min 0 and max 1
  A = np.clip(A + growth(U), 0, 1)
  img.set_array(A)
  return img,

def get_video():
    # render kernel and growth
    figure_asset(K, growth)

    # render video
    fig = figure_world(A)
    anim = matplotlib.animation.FuncAnimation(fig, evolve, frames=15, interval=2000)
    return anim.to_html5_video()

def fig_2_png(fig):
    # create IO buffer
    byte_buffer = BytesIO()

    # print raw canvas data to IO object
    fig.canvas.print_png(byte_buffer)
    img_data_str = binascii.b2a_base64(byte_buffer.getvalue())

    # create img element
    img_html = f"<img src=\"data:image/png;base64,{img_data_str}&#10;\">"
    return img_html


@app.route("/flask")
def home():
    """simplest possible example"""
    return "Hello, Flask!"

@app.route("/hello/<name>")
def hello_there(name):
    """ example with parameters
    """
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")

    # Filter the name argument to letters only using regular expressions. URL arguments
    # can contain arbitrary text, so we restrict to safe characters only.
    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + "! It's " + formatted_now
    return content

@app.route("/")
def conway():
    """Entry point for conway
    """
    html= get_video()
    #print(html)
    return html

