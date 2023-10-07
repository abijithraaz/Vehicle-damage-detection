import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
import requests
from transformers import SamModel, SamProcessor
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import cv2
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"

#Load clipseg Model
clip_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

# Load SAM model and processor
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

cache_data = None

# Prompts to segment damaged area and car
prompts = ['damaged', 'car']
damage_threshold = 0.4
vehicle_threshold = 0.5

def bbox_normalization(bbox, width, height):
    height_coeff = height/352
    width_coeff = width/352
    normalized_bbox = [int(bbox[0]*width_coeff), int(bbox[1]*height_coeff),
                       int(bbox[2]*width_coeff), int(bbox[3]*height_coeff)]
    return normalized_bbox

def bbox_area(bbox):
    area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
    return area

def segment_to_bbox(segment_indexs):
    x_points = []
    y_points = []
    for y, list_val in enumerate(segment_indexs):
        for x, val in enumerate(list_val):
            if val == 1:
                x_points.append(x)
                y_points.append(y)
    return [np.min(x_points), np.min(y_points), np.max(x_points), np.max(y_points)]

def clipseg_prediction(image):
    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
    # predict
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(1)
    # Setting threshold and classify the image contains vehicle or not
    flat_preds = torch.sigmoid(preds.squeeze()).reshape((preds.shape[0], -1))

    # Initialize a dummy "unlabeled" mask with the threshold
    flat_damage_preds_with_treshold = torch.full((2, flat_preds.shape[-1]), damage_threshold)
    flat_vehicle_preds_with_treshold = torch.full((2, flat_preds.shape[-1]), vehicle_threshold)
    flat_damage_preds_with_treshold[1:2,:] = flat_preds[0] # damage
    flat_vehicle_preds_with_treshold[1:2,:] = flat_preds[1] # vehicle

    # Get the top mask index for each pixel
    damage_inds = torch.topk(flat_damage_preds_with_treshold, 1, dim=0).indices.reshape((preds.shape[-2], preds.shape[-1]))
    vehicle_inds = torch.topk(flat_vehicle_preds_with_treshold, 1, dim=0).indices.reshape((preds.shape[-2], preds.shape[-1]))

    # bbox creation
    damage_bbox = segment_to_bbox(damage_inds)
    vehicle_bbox = segment_to_bbox(vehicle_inds)

    # Vehicle checking
    if bbox_area(vehicle_bbox) > bbox_area(damage_bbox):
        return True, bbox_normalization(damage_bbox)
    else:
        return False, []
     

@torch.no_grad()
def foward_pass(image_input: np.ndarray, points: List[List[int]]) -> np.ndarray:
    global cache_data
    image_input = Image.fromarray(image_input)
    inputs = processor(image_input, input_points=points, return_tensors="pt").to(device)
    if not cache_data or not torch.equal(inputs['pixel_values'],cache_data[0]):
        embedding = model.get_image_embeddings(inputs["pixel_values"])
        pixels = inputs["pixel_values"]
        cache_data = [pixels, embedding]
    del inputs["pixel_values"]

    outputs = model.forward(image_embeddings=cache_data[1], **inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    masks = masks[0].squeeze(0).numpy().transpose(1, 2, 0)

    return masks

def main_func(inputs):
    
    image_input = inputs['image']
    classification, points = clipseg_prediction(image_input)
    if classification:
        masks = foward_pass(image_input, points)
    
        image_input = Image.fromarray(image_input)
        
        final_mask = masks[0]
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = np.array([[128, 0, 0]])
        return Image.fromarray((mask_colors * 0.6 + image_input * 0.4).astype('uint8'), 'RGB')
    else:
        return Image.fromarray(image_input)

def reset_data():
    global cache_data
    cache_data = None

with gr.Blocks() as demo:
    gr.Markdown("# Demo to run Vehicle damage detection")
    gr.Markdown("""This app uses the SAM model and clipseg model to get a vehicle damage area from image.""")
    with gr.Row():
        image_input = gr.Image()
        image_output = gr.Image()
    
    image_button = gr.Button("Segment Image", variant='primary')

    image_button.click(main_func, inputs=image_input, outputs=image_output)
    image_input.upload(reset_data)

demo.launch()
