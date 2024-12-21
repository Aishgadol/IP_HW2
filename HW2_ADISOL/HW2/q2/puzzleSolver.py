# Yoav Simani, 208774315
# Adi Raz, 206875874

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import os
import shutil
import sys


#matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1,kp2) where kpi = (x,y)
def get_transform(matches, is_affine):
    src_points, dst_points = matches[:, 0], matches[:, 1]
    # Add your code here
    # Use the source and destination points to estimate the transform matrix
    if is_affine:
        T, _ = cv2.estimateAffine2D(src_points, dst_points)
    else:
        T, _ = cv2.findHomography(src_points, dst_points)
    return T


def stitch(image1, image2):
    # Add your code here
    # Compare each pixel in image1 and image2 and take the pixel with the max value
    stitched_image = cv2.max(image1, image2)
    return stitched_image


# Output size is (w,h)
def inverse_transform_target_image(target_img, original_transform, output_size, is_affine):
    # Add your code here
    # Find the Inverse Matrix and inverse the image accordingly
    if is_affine:
        inverse_transform = cv2.invertAffineTransform(original_transform)
        warped = cv2.warpAffine(target_img, inverse_transform, output_size, output_size, flags=cv2.INTER_LINEAR)
    else:
        inv_transform = np.linalg.inv(original_transform)
        warped = cv2.warpPerspective(target_img, inv_transform, output_size, flags=cv2.INTER_LINEAR)
    return warped


# returns list of pieces file names
def prepare_puzzle(puzzle_dir):
    edited = os.path.join(puzzle_dir, 'abs_pieces')
    if os.path.exists(edited):
        shutil.rmtree(edited)
    os.mkdir(edited)

    affine = 4 - int("affine" in puzzle_dir)

    matches_data = os.path.join(puzzle_dir, 'matches.txt')
    n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

    matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images - 1, affine, 2, 2)

    return matches, affine == 3, n_images


if __name__ == '__main__':
    lst = ['puzzle_affine_1', 'puzzle_affine_2', 'puzzle_homography_1']
    # lst = ['puzzle_affine_1']

    for puzzle_dir in lst:
        print(f'Starting {puzzle_dir}')
        puzzle = os.path.join('puzzles', puzzle_dir)
        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')
        matches, is_affine, n_images = prepare_puzzle(puzzle)
        img1 = cv2.imread(os.path.join(pieces_pth, 'piece_1.jpg'))
        height, weight, _ = img1.shape

        # To create the final puzzle we will start with image1 and then stitch the other images
        final_puzzle = img1
        for i, file_name in enumerate(os.listdir(pieces_pth)):
            # Image 1 is already part of the final puzzle so we will skip it
            if (file_name == 'piece_1.jpg'):
                continue
            T = get_transform(matches[i - 1], is_affine)
            curr_img = cv2.imread(os.path.join(pieces_pth, file_name))
            transformed_image = inverse_transform_target_image(curr_img, T, (weight, height), is_affine)

            # save the inverted image
            output_path = os.path.join(edited, f'{file_name.split('.')[0]}_relative.jpg')
            cv2.imwrite(output_path, transformed_image)

            # Stitch the inverted image with the final puzzle
            final_puzzle = stitch(final_puzzle, transformed_image)

        sol_file = f'solution.jpg'
        cv2.imwrite(os.path.join(puzzle, sol_file), final_puzzle)
