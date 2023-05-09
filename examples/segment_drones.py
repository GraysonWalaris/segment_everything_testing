import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

""" Example script for UAV detection using SAM (Segment Everything from Meta)

The bounding boxes for the drones are hardcoded, but you could easily take the 
boundingboxes from an object detection script and use them to segment the 
drones.
"""

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', 
                                facecolor=(0,0,0,0), lw=2))

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image = cv2.imread('images/drone_swarm1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = 'checkpoints/sam_vit_h.pth'
    model_type = 'vit_h'

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    predictor.set_image(image)

    input_boxes = torch.tensor([
        [375, 225, 650, 425],
        [675, 50, 900, 250],
        [80, 80 ,250, 200]
    ], device=predictor.device)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, 
                                                            image.shape[:2])
    masks, scores, logits = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    plt.axis('off')
    plt.show()

if __name__=='__main__':
    main()



