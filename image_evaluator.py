import numpy as np
from PIL import Image
from laion_aesthetics import MLP, normalizer, init_laion
import clip 
import torch
import sys

## device for the clip and aesthetic model
device = "mps" # mps -> mac m chips, can also be "cuda" or "cpu" depending on torch installation
aesthetic_model, vit_model, preprocess = init_laion(device)

### Process the prompt, only needed once!
prompt = sys.argv[2] #e.g. "sunset, bright colors"
text_inputs = clip.tokenize(prompt).to(device)
with torch.no_grad():
    text_features = vit_model.encode_text(text_inputs)
###

# Load the image from the file path
image_path = sys.argv[1] 
pil_image = Image.open(image_path)
# Display the image
pil_image.show()

## process image features
image = preprocess(pil_image).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = vit_model.encode_image(image)
im_emb_arr = normalizer(image_features.cpu().detach().numpy())

# aesthetic model prediction
prediction = aesthetic_model(torch.from_numpy(im_emb_arr).to(device).type(torch.float))

# cosine similarity between image features and text features
similarity = torch.cosine_similarity(text_features, image_features, dim=-1).mean()

aesthetic_eval_laion = prediction.item()
print("aesthetic_eval_laion", aesthetic_eval_laion)
print("similarity", similarity.item())