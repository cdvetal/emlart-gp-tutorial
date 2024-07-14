# GP https://github.com/AwardOfSky/TensorGP
from tensorgp.engine import *
# ML 
from laion_aesthetics import MLP, normalizer, init_laion
import clip 
import torch
# other needed imports
import sys
import numpy as np
from PIL import Image

## device for the clip and aesthetic model
device = "mps" # mps -> mac m chips, can also be "cuda" or "cpu" depending on torch installation
aesthetic_model, vit_model, preprocess = init_laion(device)

### Process the prompt, only needed once!
prompt = sys.argv[4] #e.g. "sunset, bright colors"
text_inputs = clip.tokenize(prompt).to(device)
with torch.no_grad():
    text_features = vit_model.encode_text(text_inputs)
###

# GP individual evaluation
def image_evaluation(**kwargs):
    # read parameters
    # get the current population
    population = kwargs.get('population')
    # the individuals have their expression based tree processed into a tensor 
    # in this case this tensor represents the image (phenotype of the individual)
    tensors = kwargs.get('tensors') 
    
    # aux code for pop evaluation
    fitnesses = []
    best_ind = 0
    ind_fitness = 0
    best_fitness = float('-inf')
    number_tensors = len(tensors)
    
    for index in range(number_tensors):
                
        # convert tensor to numpy
        image_numpy = tensors[index].numpy()
        # preprocess the image to be used in Clip image model (ViT)
        pil_image = Image.fromarray((image_numpy * 1).astype(np.uint8))
        image = preprocess(pil_image).unsqueeze(0).to(device)

        # extract the image features from the clip vit encoding
        with torch.no_grad():
            image_features = vit_model.encode_image(image)
        im_emb_arr = normalizer(image_features.cpu().detach().numpy())
        # get aesthetic prediction from the laion model
        prediction = aesthetic_model(torch.from_numpy(im_emb_arr).to(device).type(torch.float))
        aesthetic_eval_laion = prediction.item()
        
        # calculate the cosine similarity between the text features
        # of the prompt and the image features of the current individual
        similarity = torch.cosine_similarity(text_features, image_features, dim=-1).mean()
        
        ##### fitness assignment!
        # 1 - just similarity between image features and text features
        ind_fitness = similarity.item()
        
        # 2 - just aesthetics
        #ind_fitness = aesthetic_eval_laion 
        
        # 3 - combined. aesthetics is in range of values 
        # and so to compensate we will divide by a constant the aesthetic response
        #ind_fitness = similarity.item() + aesthetic_eval_laion/200.0
        
        # check and update best individual
        if ind_fitness > best_fitness:
            best_fitness = ind_fitness
            best_ind = index
        
        fitnesses.append(ind_fitness)
        population[index]['fitness'] = ind_fitness

    return population, best_ind


if __name__ == "__main__":

    # device definition for TensorGP
    dev = "gpu" #'/gpu:0'  # device to run, write '/cpu_0' to run on cpu
    # when converting from genotype to phenotype 
    # we will be creating an individual at the following resolution
    image_resolution = [224, 224, 3]
    # some EC related params
    number_generations = int(sys.argv[3])
    fset = {'abs', 'add', 'and', 'cos', 'div', 'exp', 'frac', 'if', 'log',
        'max', 'mdist', 'min', 'mult', 'neg', 'or', 'pow', 'sin', 'sqrt',  'sub', 'tan', 'warp', 'xor'}

    # multiple runs 
    for seed in range(int(sys.argv[1]),int(sys.argv[1])+int(sys.argv[2])):
        # create TensorGP engine
        engine = Engine(
                        fitness_func=image_evaluation,
                        population_size=100,
                        tournament_size=5,

                        # EC related probabilities
                        mutation_rate=0.9,
                        crossover_rate=0.5,
                        # GP specific
                        terminal_prob=0.2,
                        scalar_prob=0.50,
                        uniform_scalar_prob=1,

                        # mutations
                        max_retries=20,
                        mutation_funcs=[Engine.point_mutation, Engine.subtree_mutation, Engine.insert_mutation,
                                        Engine.delete_mutation],
                        mutation_probs=[0.25, 0.3, 0.2, 0.25],
                        min_subtree_dep=None,
                        max_subtree_dep=None,
                        
                        # tree init and depth
                        method='ramped half-and-half',
                        max_tree_depth=12,
                        min_tree_depth=-1,
                        min_init_depth=1,
                        max_init_depth=6,

                        # GP bloat control algorithm
                        bloat_control='weak', # based on the work of Silva et. al.
                        bloat_mode='depth',
                        dynamic_limit=5,
                        min_overall_size=1,
                        max_overall_size=12,

                        # function set and mapping inputs / outputs
                        domain=[-1, 1],
                        codomain=[-1, 1],
                        do_final_transform=True,
                        # directly to pixel range (8bit)
                        final_transform=[0, 255],
                        
                        # exp related
                        stop_value=number_generations,
                        effective_dims=3,
                        seed = seed,
                        operators=fset,
                        debug=0,
                        save_to_file=1,
                        save_log = True,
                        target_dims=image_resolution,
                        objective='maximizing',
                        device=dev,
                        stop_criteria='generation',
                        save_graphics=True,
                        show_graphics=False,
                        #read_init_pop_from_file="test_default.txt",
                        exp_prefix='emlart-gp',
                        save_image_pop=True,
                        save_image_best=True,
                        #image_extension="jpg",
                        image_extension="png",
                        )
        print("Evolving with the prompt:", sys.argv[4])
        engine.run()
