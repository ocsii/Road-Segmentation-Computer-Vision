To run the program

1) Open the script **RoadSegmentationAlgorithm.py** in (preferably) Spyder 

2) Navigate the "Working Directory" on the top right to the same file which contains
all the script, **images/** directory, **output/** directory, **masks/** directory, and **road_segmentation.csv** file

3) Click run

4) Two new directories with 30 images each will appear in the **output/** directory




**Program overview**

The algorithm uses an HSV colour thresholding technique, incorporating two 
crucial steps: sky / lane region detection and largest connected component detection. 

Only focusing on the road will cause many false positives. The algorithm also ensures the lane markings are 
excluded from the road mask. 

Compared to a set of ground truth masks (in the **masks/** directory) the algorithm achieves an accuracy of 96.2% (from IOU)



**Example input: **
![0](https://github.com/user-attachments/assets/e63f7279-31b0-456a-b900-70abf544dc9f)

**Example output:**
Output 1: Base segmented images
![0](https://github.com/user-attachments/assets/94db04e4-35fd-4edb-a517-40481692a588)

Output 2: Combination of outputs (original image, segmented image, masks, IOU)
![0](https://github.com/user-attachments/assets/24472b23-9b79-466c-85d2-4c8d41587208)
