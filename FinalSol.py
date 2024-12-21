# Idan Morad, 316451012
# Nadav Melman, 206171548

import cv2
import numpy as np
import os
import shutil
import sys


# matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1,kp2) where kpi = (x,y)
def get_transform(matches, is_affine):
    # use src/dst coordinates to estimate transformations
    src_cords=matches[:, 0]
    dst_cords=matches[:, 1]
    if not is_affine:
        T, _ = cv2.findHomography(src_cords, dst_cords)
    else:
        T, _ = cv2.estimateAffine2D(src_cords, dst_cords)

    return T


def stitch(img1, img2):
    #find max value of each pixel in the images
    combined = cv2.max(img1, img2)
    return combined



def inverse_transform_target_image(target_img, original_transform, output_size):
    # determine transform type by matrix shape, invert accordingly
    if not original_transform.shape == (2, 3):
        # homography case
        inverse_h = np.linalg.inv(original_transform)
        result = cv2.warpPerspective(target_img, inverse_h, output_size, flags=cv2.INTER_LINEAR)
    else:
        # affine case
        inverse_aff = cv2.invertAffineTransform(original_transform)
        result = cv2.warpAffine(target_img, inverse_aff, output_size, output_size, flags=cv2.INTER_LINEAR)
    return result


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

    for puzzle_dir in lst:
        print(f'Starting {puzzle_dir}')

        puzzle = os.path.join('puzzles', puzzle_dir)

        pieces_path = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')

        matches, is_affine, n_images = prepare_puzzle(puzzle)

        # Add your code here
        #load piece1 which is already in correct position
        piece1 = cv2.imread(os.path.join(pieces_path, 'piece_1.jpg'))
        h, w, _ = piece1.shape
        #put first piece in final puzzle and iterate over rest of image pieces
        final_puzzle = piece1

        all_pieces = os.listdir(pieces_path)
        for i, filename in enumerate(all_pieces):
            #if we're looking at the first peice (which we already used) just skip iteraton
            if filename == 'piece_1.jpg':
                continue

            # get transform to place piece onto the final puzzle
            transform = get_transform(matches[i - 1], is_affine)

            # load current piece and inverse transform it
            curr_image = cv2.imread(os.path.join(pieces_path, filename))
            inverse_piece = inverse_transform_target_image(curr_image, transform, (w, h))

            # save inverted image
            outpath = os.path.join(edited, f'{filename.split(".")[0]}_relative.jpg')
            cv2.imwrite(outpath, inverse_piece)

            # stitch it into the puzzle
            final_puzzle = stitch(final_puzzle, inverse_piece)
        # End of "Add your code here"

        sol_file = f"solution.jpg"
        cv2.imwrite(os.path.join(puzzle, sol_file), final_puzzle)
