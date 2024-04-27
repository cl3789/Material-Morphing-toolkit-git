import numpy as np
import copy
import torch
import PIL.Image



def generate_from_w(G, w_codes, noise_mode = "const"):
  """Generate images from StyleGAN's latent codes (either W or W+)"""
  device = torch.device('cuda')
  w_codes = torch.tensor(w_codes, device=device)
  assert w_codes.shape[1:] == (G.num_ws, G.w_dim)

  generated = []
  for _, w in enumerate(w_codes):
      img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode, force_fp32=True)
      img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
      img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
      generated.append(img)
  return generated


def edit_layer(w_codes, direction, step, layer_indices, start_distance, end_distance):
  """Linear interpolate the latent codes (either W or W+) at selected layers (layer_indices)"""

  x = w_codes[:, np.newaxis]

  results = np.tile(x, [step if axis == 1 else 1 for axis in range(x.ndim)])

  is_manipulatable = np.zeros(results.shape, dtype=bool)

  distance = np.linspace(start_distance, end_distance, step)
  print(distance)
  l = distance.reshape(
      [step if axis == 1 else 1 for axis in range(x.ndim)])

  is_manipulatable[:, :, layer_indices] = True
  results = np.where(is_manipulatable,  x + l * direction, results)

  return results


def linear_interpolate(a,b,t):
     return a + (b - a) * t


class MergeDomains():
  def __init__(self, G_source, G_target, steps):
    self.G1 = copy.deepcopy(G_source) # StyleGAN generator of source material
    self.G2 = copy.deepcopy(G_target) # StyleGAN generator of target material
    self.steps = steps # number of morphing steps

  def get_synthesis_state_dict(self, G):
    state_dict = copy.deepcopy(G.synthesis.state_dict())
    return state_dict

  def get_weights_names(self, layer_names):
    state_dict = self.get_synthesis_state_dict(self.G1)
    names = list(state_dict.keys())
    wnames = []

    for name in names:
      wnames.append(name)

    if layer_names == "all": ## Morphing ALL 18 layers
      grab_layers = wnames

    elif layer_names == "4to16": ## Morphing 4x4 res to 16x16 res
      grab_layers = wnames[0:51]

    elif layer_names == "4to32": ## Morphing 4x4 res to 32x32 res
      grab_layers = wnames[0:70]

    elif layer_names == "4to64": ## Morphing 4x4 res to 64x64 res
      grab_layers = wnames[0:89]

    elif layer_names == "4to128": ## Morphing 4x4 res to 128x128 res
      grab_layers = wnames[0:108]

    elif layer_names == "32res": ## Morphing 32x32 res
      grab_layers = wnames[51:70] 

    elif layer_names == "32to64": ## Morphing 32x32 res to 64x64 res
      grab_layers = wnames[51:89] 

    elif layer_names == "32to128": ## Morphing 32x32 res to 128x128 res
      grab_layers = wnames[51:108] 

    elif layer_names == "64to128": ## Morphing 64x64 res to 128x128 res
      grab_layers = wnames[70:108]

    elif layer_names == "16up": ## Morphing 16x16 res to 1024x1024 res
      grab_layers = wnames[21:]

    elif layer_names == "32up":
      grab_layers = wnames[51:] ## Morphing 32x32 res to 1024x1024 res

    elif layer_names == "64res":
      grab_layers = wnames[70:89] ## Morphing 64x64 res

    elif layer_names == "64up":
      grab_layers = wnames[70:] ## Morphing 64x64 res to 1024x1024 res

    elif layer_names == "128up":
      grab_layers = wnames[89:] ## Morphing 128x128 res to 1024x1024 res

    elif layer_names == "32to512":
      grab_layers = wnames[51:146] ## Morphing 32x32 res to 512x512 res

    elif layer_names == "64to128_with_1024":
      grab_layers = wnames[70:108]
      grab_layers = grab_layers + wnames[146:]

    else:
      print("invalid")
    print(grab_layers)
    return grab_layers

  def get_morph_generator(self, layers):
    """Produce a list of StyleGAN generators with morphed weights at selected layers (layers)"""
    
    distance = np.linspace(0, 1, self.steps)
    print("morphing steps:",distance)
    all_layers = self.get_weights_names("all")
    subset_layers = self.get_weights_names(layers)

    G_models = [] # initialize a list for saving interpolated generators
    for t in distance:
      G_new = copy.deepcopy(self.G1)
      G_new_state_dict = {}
      for name in all_layers:
        from_weights = self.get_synthesis_state_dict(self.G1)[name]
        to_weights = self.get_synthesis_state_dict(self.G2)[name]
        if name in subset_layers:
          G_new_state_dict[name] = linear_interpolate(from_weights, to_weights, t)
        else:
          G_new_state_dict[name] = from_weights
      G_new.synthesis.load_state_dict(G_new_state_dict)
      G_models.append(G_new)
    return G_models

  def morph_domain_layer(self, w_source, w_target, G_models,
                         layers_ind=list(range(18)), mode = "1D"):
    """
    Morph latent code from w_source to w_target at selected layers (layers_ind)
    w_source: source image latent code in shape (18, 512),
    w_target: target image latent code in shape (18, 512)
    """
    direction = w_target - w_source
    res = edit_layer(np.expand_dims(w_source, axis=0), direction = direction, step=self.steps, layer_indices = layers_ind, start_distance=0, end_distance=1)

    print("Number of Generators:",len(G_models))

    res_imgs = []
    if mode == "1D":
      for i in range(self.steps):
        res_img = generate_from_w(G_models[i], np.expand_dims(res[0][i], axis = 0))
        res_imgs.append(res_img[0])
    elif mode == "Grid":
      for i in range(self.steps):
        for j in range(self.steps):
          res_img = generate_from_w(G_models[i], np.expand_dims(res[0][j], axis = 0))
          res_imgs.append(res_img[0])

    return res_imgs

  def get_midpoint(self, w_source, w_target, G_layers, layers_ind=list(range(18))):
    direction = w_target - w_source
    num_img = w_target.shape[0]
    # Set the step=3 for the midpoint for the interpolation of the latent codes
    res = edit_layer(np.expand_dims(w_source, axis=0), direction = direction, step=3, layer_indices = layers_ind, start_distance=0, end_distance=1)
    new_codes = res[0][1]

    G_new = self.get_morph_generator(G_layers)[1]

    res_imgs = []
    for i in range(num_img):
      img = generate_from_w(G_new, np.expand_dims(new_codes[i], axis = 0))
      res_imgs.append(img[0])
    return res_imgs

  def switch_layer(self, w_source, w_target, G_layers, layers_ind=list(range(18))):
    direction = w_target - w_source
    num_img = w_target.shape[0]
    res = edit_layer(np.expand_dims(w_source, axis=0), direction = direction, step=2, layer_indices = layers_ind, start_distance=0, end_distance=1)
    new_codes = res[0][1]

    G_new = self.get_morph_generator(G_layers)[-1]

    res_imgs = []
    for i in range(num_img):
      img = generate_from_w(G_new, np.expand_dims(new_codes[i], axis = 0))
      res_imgs.append(img[0])
    return res_imgs
