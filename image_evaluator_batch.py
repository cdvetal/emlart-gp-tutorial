import pandas as pd
import numpy as np
from PIL import Image
from laion_aesthetics import MLP, normalizer, init_laion
import clip 
import torch
import sys
from pathlib import Path

def evaluate_images_in_folder(folder_path, prompt, device="cpu"):
    aesthetic_model, vit_model, preprocess = init_laion(device)

    # Encode prompt
    text_inputs = clip.tokenize(prompt).to(device)
    with torch.no_grad():
        text_features = vit_model.encode_text(text_inputs)

    
    
    
    results = []

    # Iterate over image files
    folder = Path(folder_path)
    for image_path in folder.glob("*"):
        if not image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        try:
            pil_image = Image.open(image_path).convert("RGB")
            image = preprocess(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = vit_model.encode_image(image)

            im_emb_arr = normalizer(image_features.cpu().detach().numpy())

            # aesthetic model prediction
            prediction = aesthetic_model(torch.from_numpy(im_emb_arr).to(device).type(torch.float))

            # cosine similarity between image features and text features
            similarity = torch.cosine_similarity(text_features, image_features, dim=-1).mean()
            
            aesthetic_eval_laion = prediction.item()
            similarity_score = similarity.item()
                    
            
            print(f"{image_path.name} - Aesthetic: {aesthetic_eval_laion:.4f}, Similarity: {similarity_score:.4f}")

            results.append({
                "image": image_path.name,
                "aesthetic_score_laion": aesthetic_eval_laion,
                "similarity_score": similarity_score
            })

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python image_evaluator_batch.py <mps/cuda/cpu> <image_folder> <prompt>")
        sys.exit(1)

    device = sys.argv[1] # "mps"  # or "cuda", or "cpu"
    image_folder = sys.argv[2]
    prompt = sys.argv[3]
    
    df = evaluate_images_in_folder(image_folder, prompt, device)
    df.to_csv("evaluation_results.csv", index=False)
    print("Results exported to evaluation_results.csv")
