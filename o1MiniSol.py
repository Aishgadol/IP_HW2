# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import os
import shutil

# matches is of (3|4 x 2 x 2) size. each row is a match - pair of (kp1, kp2) where kpi = (x, y)
def get_transform(matches, is_affine):
    source_pts, destination_pts = matches[:, 0], matches[:, 1]
    print(f"Source points shape: {source_pts.shape}")
    print(f"Destination points shape: {destination_pts.shape}")

    # check if there are enough points
    if is_affine and source_pts.shape[0] < 3:
        print("Error: Not enough points for affine transformation. At least 3 points are required.")
        return None
    if not is_affine and source_pts.shape[0] < 4:
        print("Error: Not enough points for homography. At least 4 points are required.")
        return None

    # estimate transformation matrix based on type
    if is_affine:
        transform_matrix, inliers = cv2.estimateAffine2D(source_pts, destination_pts)
    else:
        transform_matrix, inliers = cv2.findHomography(source_pts, destination_pts)

    print(f"Transform matrix:\n{transform_matrix}")
    return transform_matrix

def stitch_images(base_image, new_image):
    # merge two images by taking the maximum pixel value at each position
    stitched_image = cv2.max(base_image, new_image)
    return stitched_image

# output size is (width, height)
def inverse_transform_target_image(target_image, transform_matrix, output_size, is_affine):
    # compute inverse transformation and warp the image
    if is_affine:
        try:
            inverse_matrix = cv2.invertAffineTransform(transform_matrix)
            warped_image = cv2.warpAffine(target_image, inverse_matrix, output_size, flags=cv2.INTER_LINEAR)
        except cv2.error as e:
            print(f"Error in affine transformation: {e}")
            return None
    else:
        try:
            inverse_matrix = np.linalg.inv(transform_matrix)
            warped_image = cv2.warpPerspective(target_image, inverse_matrix, output_size, flags=cv2.INTER_LINEAR)
        except np.linalg.LinAlgError as e:
            print(f"Error in homography inversion: {e}")
            return None
    return warped_image

# returns list of piece file names
def prepare_puzzle(puzzle_dir):
    abs_pieces_dir = os.path.join(puzzle_dir, 'abs_pieces')
    if os.path.exists(abs_pieces_dir):
        shutil.rmtree(abs_pieces_dir)
    os.mkdir(abs_pieces_dir)

    # determine if the transform is affine or homography based on folder name
    is_affine = "affine" in puzzle_dir
    transform_order = 3 if is_affine else 4  # 3 for affine, 4 for homography

    matches_file = os.path.join(puzzle_dir, 'matches.txt')
    pieces_dir = os.path.join(puzzle_dir, 'pieces')
    num_pieces = len(os.listdir(pieces_dir))

    try:
        # load matches as float for accurate transformation
        matches = np.loadtxt(matches_file, dtype=np.float32)
        print(f"Total matches loaded: {matches.shape[0]}")

        expected_matches = (num_pieces - 1) * transform_order
        if matches.shape[0] != expected_matches:
            print(f"Warning: Expected {expected_matches} matches, but found {matches.shape[0]}")

        matches = matches.reshape(num_pieces - 1, transform_order, 2, 2)
        print(f"Matches reshaped to: {matches.shape}")
    except Exception as e:
        print(f"Error loading matches from {matches_file}: {e}")
        return None, None, None

    return matches, is_affine, num_pieces


if __name__ == '__main__':
    # list of puzzle directories to process
    puzzle_list = ['puzzle_affine_1']  # add more puzzles as needed

    for puzzle_name in puzzle_list:
        print(f'starting {puzzle_name}')

        puzzle_path = os.path.join('puzzles', puzzle_name)
        pieces_path = os.path.join(puzzle_path, 'pieces')
        abs_pieces_path = os.path.join(puzzle_path, 'abs_pieces')

        matches, is_affine, total_pieces = prepare_puzzle(puzzle_path)

        if matches is None:
            print(f'skipping {puzzle_name} due to setup errors.')
            continue

        # load the first puzzle piece which is already correctly placed
        base_piece_path = os.path.join(pieces_path, 'piece_1.jpg')
        base_image = cv2.imread(base_piece_path)
        if base_image is None:
            print(f'failed to load {base_piece_path}')
            continue
        height, width, _ = base_image.shape

        # initialize the final puzzle image with the first piece
        final_puzzle = base_image.copy()

        # iterate through all puzzle pieces
        for idx, piece_file in enumerate(sorted(os.listdir(pieces_path))):
            # skip the first piece as it's already in the final puzzle
            if piece_file == 'piece_1.jpg':
                continue

            piece_path = os.path.join(pieces_path, piece_file)
            current_piece = cv2.imread(piece_path)
            if current_piece is None:
                print(f'failed to load {piece_path}')
                continue

            # get the corresponding match
            match_idx = idx - 1
            if match_idx < 0 or match_idx >= matches.shape[0]:
                print(f'no match data for {piece_file}')
                continue
            transform_params = matches[match_idx]

            # compute transformation matrix
            transform_matrix = get_transform(transform_params, is_affine)
            if transform_matrix is None:
                print(f'failed to compute transformation for {piece_file}')
                continue

            # apply inverse transformation to align the piece
            aligned_image = inverse_transform_target_image(current_piece, transform_matrix, (width, height), is_affine)
            if aligned_image is None:
                print(f'failed to align {piece_file}')
                continue

            # save the aligned piece
            aligned_filename = f"{os.path.splitext(piece_file)[0]}_abs.jpg"
            aligned_path = os.path.join(abs_pieces_path, aligned_filename)
            cv2.imwrite(aligned_path, aligned_image)
            print(f'saved aligned piece to {aligned_path}')

            # merge the aligned image with the final puzzle
            final_puzzle = stitch_images(final_puzzle, aligned_image)

        # save the completed puzzle image
        solution_path = os.path.join(puzzle_path, 'solution.jpg')
        cv2.imwrite(solution_path, final_puzzle)
        print(f'saved solution to {solution_path}')
