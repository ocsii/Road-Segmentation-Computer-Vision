import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as pt
import matplotlib
import time

# Prevent figures from popping up when combining images 
matplotlib.use('Agg') #

# Load each image
def load_image(image_path):
    return cv2.imread(image_path)


# Convert mask to correct colours and load
def load_mask(mask_path):
    mask = cv2.imread(mask_path)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)


# Save each image
def save_image(output_path, image):
    
    # Set output directory
    output_dir = os.path.dirname(output_path)
    
    # Create directory if does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cv2.imwrite(output_path, image)
    
    if not os.path.exists(output_path):
        print(f"Failed to save image: {output_path}\n")


# Filter the image / Blur the image to reduce noise
def process_filter(image):
    return cv2.GaussianBlur(image, (5,5), 0)


# Convert image to HSV
def process_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# Create mask for road region
def create_road_mask(hsv):
    
    # Set gray thresholds to sample road colour
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 200])
    
    # Use thresholds to extract road
    road_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Perform closing (remove gaps) then opening
    kernel = np.ones((5, 5), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
    return road_mask


# Create mask for sky region
def create_sky_mask(hsv):
    
    # Set threshold to sample sky
    lower_sky = np.array([90, 0, 150])
    upper_sky = np.array([135, 255, 255])
    
    # Extract sky from thresholds
    sky_mask = cv2.inRange(hsv, lower_sky, upper_sky)
    
    # Perform closing (remove gaps) then opening
    kernel = np.ones((5, 5), np.uint8)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
    return sky_mask


# Create mask for the lane lines and sky region (again)
def create_lane_and_sky_mask(image):
    
    # Define the range for white color (lane) in RGB
    lower_white = np.array([158, 158, 158])
    upper_white = np.array([255, 255, 255])

    # Create a binary mask where white colors are white and the rest are black
    mask_white = cv2.inRange(image, lower_white, upper_white)

    # Perform morphological operations to refine the mask for white
    kernel = np.ones((4, 4), np.uint8)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask_white  # Return the binary mask


# Subtract masks
def subtract_masks(mask1, mask2):
    mask1 = cv2.subtract(mask1, mask2)

    return mask1


# Keep only largest connected component assumed to be the road
# Remove any small isolated regions that is not road
def get_largest_connected_component(mask):
    num_labels, labels_im = cv2.connectedComponents(mask)
    
    # Get label for largest connected component
    largest_component = 1 + np.argmax(np.bincount(labels_im.flat)[1:])
    
    # Create final mask with largest connected component
    final_mask = np.uint8(labels_im == largest_component) * 255
    return final_mask


# Highlight the road in the image
def highlight_road(image, final_mask):
    highlighted_road = image.copy()
    highlighted_road[final_mask == 255] = [255, 0, 0]
    return highlighted_road


# Extract the road region and create a binary mask of the road region
# Used to get the IOU of (road region) only of segmented image and provided mask
def extract_road_region(image):

    # Convert the image to HSV 
    hsv_mask = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Set blue thresholds that represent road region
    lower_blue = np.array([90, 50, 50]) 
    upper_blue = np.array([130, 255, 255]) 
    
    # Create mask 
    road_mask = cv2.inRange(hsv_mask, lower_blue, upper_blue)
    road_mask[road_mask > 0] = 255
    
    return road_mask


# Calculate IOU of extracted image with provided mask (ground truths)
# Takes
# ground_truth_mask - the provided mask as ground truth
# segmented_image   - the output from the segmentation algorithm
def calculate_iou(road_extracted_ground_truth_mask, road_extracted_segmented_image):
    
    # Calulate IOU
    intersection = np.logical_and(road_extracted_ground_truth_mask, road_extracted_segmented_image)
    union = np.logical_or(road_extracted_ground_truth_mask, road_extracted_segmented_image)
    iou = np.sum(intersection) / np.sum(union)
    
    print(f"IOU with provided mask: {iou:.4f}")
    
    return iou


# Main segmentation algorithm
def segment_road(image):
    
    # Remove noise
    blur = process_filter(image)
    
    # Convert to HSV
    hsv = process_hsv(blur)
    
    # Create road masks and sky masks and lane mask
    road_mask = create_road_mask(hsv)
    sky_mask = create_sky_mask(hsv)
    lane_mask = create_lane_and_sky_mask(image)
    
    # Remove sky region 1 from road mask
    road_mask = subtract_masks(road_mask, sky_mask)
    
    # Remove lane lines and sky region 2 from road mask
    road_mask = subtract_masks(road_mask, lane_mask)

    # Get largest connected component (remove small / stray parts of the mask)
    final_mask = get_largest_connected_component(road_mask)
    
    # Highlight road
    highlighted_road = highlight_road(image, final_mask)
    
    
    return highlighted_road


def main():
    # Read CSV file
    csv_file = 'roads_segmentation.csv'
    df = pd.read_csv(csv_file)
    
    # Create output directories if they do not exist
    output_dir = 'output'
    
    segmented_output_dir = os.path.join(output_dir, 'Segmented Images')
    combined_output_dir = os.path.join(output_dir, 'Combined Output')
    
    for dir in [output_dir, segmented_output_dir, combined_output_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    # Array to store and calculate IOU average
    ious = []
    times = []    
    
    for idx, row in df.iterrows():
        
        # ============== Path setting
        
        # Get image path and image name (without the images/ )
        current_image_path = row['image_name']
        current_image_name = current_image_path.split('/')[-1]
        
        # Set output paths
        segmented_output_dir_path = os.path.join(segmented_output_dir, current_image_name)
        combined_output_dir_path = os.path.join(combined_output_dir, current_image_name)
        
        
        # ============== Segmentation
        
        # Start timer
        start_time = time.time()
        
        # Load and segment image
        image = load_image(current_image_path)
        segmented_image = segment_road(image)
        
        # End timer
        end_time = time.time()
        
        # Calculate time taken
        time_taken = end_time - start_time
        times.append(time_taken)
        
        # Save image
        save_image(segmented_output_dir_path, segmented_image)

        print(f"\nSegmented and saved image: {current_image_name}")
        print(f"Time taken: {time_taken:.4f}")


        # ============== Binary mask extraction
        
        # Set ground truth mask path and load
        mask_path = os.path.join('masks', current_image_name)
        ground_truth_mask = load_image(mask_path)
        
        # Create binary mask of road regions for IOU calculation
        road_extracted_ground_truth_mask = extract_road_region(ground_truth_mask)
        road_extracted_segmented_image = extract_road_region(segmented_image)
        
        
        # ============== IOU calculation
        
        # Calculate IOU and get the extracted road regions
        iou = calculate_iou(road_extracted_ground_truth_mask, road_extracted_segmented_image)
        ious.append(iou)
    
        
        # ============== Combining images for output
        
        # Combine the masks and save
        fig, axs = pt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title('Original Image')
        axs[0, 1].imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title('Segmented Image')
        axs[1, 0].imshow(road_extracted_ground_truth_mask, cmap='gray')
        axs[1, 0].set_title('Ground Truth Mask (From Provided Masks)')
        axs[1, 1].imshow(road_extracted_segmented_image, cmap='gray')
        axs[1, 1].set_title('Segmented Mask (From Algorithm Output)')
        
        # Add IOU text below the mask images
        fig.text(0.5, 0.02, f'IOU from Masks: {iou:.4f}', ha='center', fontsize=12)
        
        for ax in axs.flatten():
            ax.axis('off')
        
        fig.savefig(combined_output_dir_path)
        
     
    # Calculate average IOU
    avg_iou = np.mean(ious)
    print(f"\nAverage IOU: {avg_iou:.4f}")
    
    # Calulate avrage processing time
    avg_time = np.mean(times)
    print(f"Average time per image: {avg_time:.4f}")


# 20045472 - Christian Finta Cham
if __name__ == "__main__":
    main()

