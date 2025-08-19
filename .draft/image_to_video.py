import cv2
import os
import argparse

def create_video_from_images(folder_path):
    # List all PNG files in the folder
    images = [img for img in os.listdir(folder_path) if img.endswith(".png")]
    
    if not images:
        print(f"No PNG images found in {folder_path}")
        return

    # Sort images by name (or by any other criteria you need)
    try:
        images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    except ValueError:
        images.sort()  # Fallback to alphabetical sort if numeric sort fails
    
    # Read the first image to get the frame size
    frame = cv2.imread(os.path.join(folder_path, images[0]))
    if frame is None:
        print(f"Could not read the first image: {images[0]}")
        return
    
    height, width, layers = frame.shape

    # Create a video writer object
    # video_path = os.path.join(folder_path, os.path.basename(folder_path) + '.mp4')
    video_path = folder_path + '.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID', 'MJPG', etc.
    video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))

    # Read each image and write it to the video
    for i, image in enumerate(images):
        image_path = os.path.join(folder_path, image)
        img = cv2.imread(image_path)
        if img is not None:
            video.write(img)
        else:
            print(f"Warning: Could not read image {image}")

    # Release the video writer to finalize the video file
    video.release()

    print(f"Video saved as {video_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert a folder of PNG images to a video.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing PNG images')
    
    args = parser.parse_args()
    
    # Validate the folder path exists
    if not os.path.isdir(args.folder_path):
        print(f"Error: The specified path does not exist or is not a directory: {args.folder_path}")
    else:
        create_video_from_images(args.folder_path)
