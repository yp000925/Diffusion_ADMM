import diffusers
from diffusers import UNet2DModel,DDPMPipeline
import json

## ------The default model----- ##
# model = UNet2DModel()
# with open("modelconfig.json",'w') as write_f:
#     json.dump(config, write_f,indent=4)


## ---- Load model from config with random initialization ----- ##
config = json.load(open("modelconfig.json"))
model = UNet2DModel(**config)

## ---- Load model from pre-trained weights and configuration ----- ##
# repo_id = "google/ddpm-church-256"
# model = UNet2DModel.from_pretrained(repo_id)

## ---- Save model with config and weights ----- ##
model.save_pretrained("my_model") # this will generate the json file and bin file for weights