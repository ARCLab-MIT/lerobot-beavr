import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2

class ObjectDescriptor:
    """
    Provides object description capabilities using OpenAI's CLIP model.
    
    This class can classify objects using CLIP's zero-shot capabilities by comparing
    image embeddings with text embeddings of candidate labels.
    """
    
    def __init__(self, method="clip", device=None, debug=False):
        """
        Initialize the object descriptor.
        
        Args:
            method (str): Method to use for description (only 'clip' is supported)
            device (str): Device to use for computation ('cuda' or 'cpu')
            debug (bool): Whether to print debug information
        """
        self.method = method
        self.debug = debug
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if self.debug:
            print(f"Using device: {self.device}")
            
        # Load CLIP model and processor
        model_name = "openai/clip-vit-large-patch14"
        if self.debug:
            print(f"Loading CLIP model: {model_name}")
            
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Default set of common object categories aligned with YOLOv8 detection capabilities
        self.default_labels = [
            "human", "robot", "gear", "bin", "floor", "wall", "table", "box",
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
            "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush", "mechanical part", "electronic device", "tool"
        ]
        
        if self.debug:
            print("CLIP model loaded successfully")
    
    def describe(self, image, mask=None, candidate_labels=None):
        """
        Describe the object in the image.
        
        Args:
            image: PIL Image or numpy array
            mask (numpy.ndarray, optional): Binary mask for the object
            candidate_labels (list): List of candidate labels to choose from
            
        Returns:
            dict: Dictionary containing the results
        """
        if candidate_labels is None:
            # Use the comprehensive default set of object categories
            candidate_labels = self.default_labels
        
        # Apply mask if provided
        if mask is not None and isinstance(image, np.ndarray):
            # Create a masked image
            masked_image = image.copy()
            if len(mask.shape) == 2:  # Binary mask
                # Apply mask to each channel
                for c in range(3):
                    masked_image[:, :, c] = masked_image[:, :, c] * mask
            image = masked_image
        
        # Convert image to PIL if it's a numpy array
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image.astype('uint8'))
        
        # Prepare text inputs
        text_inputs = [f"a photo of a {label}" for label in candidate_labels]
        
        # Process inputs
        inputs = self.processor(
            text=text_inputs,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get similarity scores
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0].cpu().numpy()
        
        # Create results dictionary
        results = {
            "labels": candidate_labels,
            "scores": probs.tolist()
        }
        
        # Get the most likely label
        best_label_idx = np.argmax(probs)
        results["best_label"] = candidate_labels[best_label_idx]
        results["best_score"] = float(probs[best_label_idx])
        
        # Get top-5 labels and scores
        top_indices = np.argsort(probs)[::-1][:5]
        results["top_labels"] = [candidate_labels[i] for i in top_indices]
        results["top_scores"] = [float(probs[i]) for i in top_indices]
        
        if self.debug:
            print(f"Best label: {results['best_label']} with score {results['best_score']:.4f}")
            print(f"Top 5 labels: {results['top_labels']}")
            print(f"Top 5 scores: {[f'{score:.4f}' for score in results['top_scores']]}")
            
        return results