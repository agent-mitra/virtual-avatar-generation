import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

'''

Can modify these (below are SAM's default values):

class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32, #32, #maybe try 1024
        points_per_batch: int = 1024, #64,
        pred_iou_thresh: float = 0.7, #0.8, 
        stability_score_thresh: float = 0.92,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 1, #was 1
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512/1500, #0.1, #512 / 1500,
        crop_n_points_downscale_factor: int = 1, #1 or 2,
        point_grids: Optional[List[np.ndarray]] = None, 
        min_mask_region_area: int = 5, #cleaner masks with 10, #was 0
        output_mode: str = "binary_mask",
        
        
Notes:
box_nms_thresh: can remove duplicate mask by their bounding box iou
crop_n_layers=1, and crop_n_points_downscale_factor=2, can get you finer results because it use multi crops to extract features and decode masks
min_mask_resion_area can remove "holes" and "islands" attached to every single mask

'''

class Segmentor:
    def __init__(self, sam_args):
        """
        sam_args:
            sam_checkpoint: path of SAM checkpoint
            generator_args: args for everything_generator
            gpu_id: device
        """
        self.device = sam_args["gpu_id"]
        self.sam = sam_model_registry[sam_args["model_type"]](checkpoint=sam_args["sam_checkpoint"])
        self.sam.to(device=self.device)
        self.everything_generator = SamAutomaticMaskGenerator(self.sam, points_per_side=64, points_per_batch=2048) #**sam_args['generator_args'], points_per_side=64
        self.interactive_predictor = self.everything_generator.predictor
        self.have_embedded = False
        
    @torch.no_grad()
    def set_image(self, image):
        # calculate the embedding only once per frame.
        if not self.have_embedded:
            self.interactive_predictor.set_image(image)
            self.have_embedded = True
    @torch.no_grad()
    def interactive_predict(self, prompts, mode, multimask=True):
        assert self.have_embedded, 'image embedding for sam need be set before predict.'        
        
        if mode == 'point':
            masks, scores, logits = self.interactive_predictor.predict(point_coords=prompts['point_coords'], 
                                point_labels=prompts['point_modes'], 
                                multimask_output=multimask)
        elif mode == 'mask':
            masks, scores, logits = self.interactive_predictor.predict(mask_input=prompts['mask_prompt'], 
                                multimask_output=multimask)
        elif mode == 'point_mask':
            masks, scores, logits = self.interactive_predictor.predict(point_coords=prompts['point_coords'], 
                                point_labels=prompts['point_modes'], 
                                mask_input=prompts['mask_prompt'], 
                                multimask_output=multimask)
                                
        return masks, scores, logits
        
    @torch.no_grad()
    def segment_with_click(self, origin_frame, coords, modes, multimask=True):
        '''
            
            return: 
                mask: one-hot 
        '''
        self.set_image(origin_frame)

        prompts = {
            'point_coords': coords,
            'point_modes': modes,
        }
        masks, scores, logits = self.interactive_predict(prompts, 'point', multimask)
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        prompts = {
            'point_coords': coords,
            'point_modes': modes,
            'mask_prompt': logit[None, :, :]
        }
        masks, scores, logits = self.interactive_predict(prompts, 'point_mask', multimask)
        mask = masks[np.argmax(scores)]

        return mask.astype(np.uint8)

    def segment_with_box(self, origin_frame, bbox, reset_image=False):
        if reset_image:
            self.interactive_predictor.set_image(origin_frame)
        else:
            self.set_image(origin_frame)
        # coord = np.array([[int((bbox[1][0] - bbox[0][0]) / 2.),  int((bbox[1][1] - bbox[0][1]) / 2)]])
        # point_label = np.array([1])

        masks, scores, logits = self.interactive_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]),
            multimask_output=True
        )
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]

        masks, scores, logits = self.interactive_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([[bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]]),
            mask_input=logit[None, :, :],
            multimask_output=True
        )
        mask = masks[np.argmax(scores)]
        
        return [mask]
