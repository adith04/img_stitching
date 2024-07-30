#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import message_filters
import concurrent.futures

bits_map = []

class ImageStitcher:
    def __init__(self):
        self.bridge = CvBridge()
        self.images = {
            'left': None,
            'front': None,
            'right': None,
            'up': None,
            'down': None
        }
        self.lock = threading.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        left_sub = message_filters.Subscriber('webcam_left_front/image_raw_left_front', Image, queue_size=1)
        front_sub = message_filters.Subscriber('webcam_front/image_raw_front', Image, queue_size=1)
        right_sub = message_filters.Subscriber('webcam_right_front/image_raw_right_front', Image, queue_size=1)
        up_sub = message_filters.Subscriber('webcam_up/image_raw_up', Image, queue_size=1)
        down_sub = message_filters.Subscriber('webcam_down/image_raw_down', Image, queue_size=1)

        ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, front_sub, right_sub, up_sub, down_sub], queue_size=1, slop=0.1
        )
        ts.registerCallback(self.image_callback)

        self.pub = rospy.Publisher('stitched_image', Image, queue_size=1)
        self.stitching_thread = threading.Thread(target=self.stitch_and_publish)
        self.stitching_thread.daemon = True
        self.stitching_thread.start()

    def image_callback(self, left, front, right, up, down):
        with self.lock:
            self.images['left'] = self.bridge.imgmsg_to_cv2(left, "bgr8")
            self.images['front'] = self.bridge.imgmsg_to_cv2(front, "bgr8")
            self.images['right'] = self.bridge.imgmsg_to_cv2(right, "bgr8")
            self.images['up'] = self.bridge.imgmsg_to_cv2(up, "bgr8")
            self.images['down'] = self.bridge.imgmsg_to_cv2(down, "bgr8")

    def stitch_and_publish(self):
        rate = rospy.Rate(5)  # Adjusted the rate to 10 Hz
        while not rospy.is_shutdown():
            with self.lock:
                if all(image is not None for image in self.images.values()):
                    future = self.executor.submit(self.stitch_images)
                    stitched_image = future.result()
                    if stitched_image is not None:
                        stitched_msg = self.bridge.cv2_to_imgmsg(stitched_image, 'bgr8')
                        self.pub.publish(stitched_msg)
                        # Reset images to avoid re-stitching the same set
                        self.images = {key: None for key in self.images}
            rate.sleep()

    def stitch_images(self):
        img_left = self.images['left']
        img_front = self.images['front']
        img_right = self.images['right']
        img_up = self.images['up']
        img_down = self.images['down']

        try:
            upH = np.array([[ 8.77117773e-01, -1.96206979e-01, 6.11303304e+01],
                            [-5.99665375e-04, 7.04209450e-01, 1.68395999e+02],
                            [-9.20787802e-07, -3.92825216e-04, 1.00000000e+00]])
            up_stitch = Combine_Images(img_down, img_front, preCompH=upH)
            dH = np.array([[1.25065679e+00, 3.49143801e-01, -3.98029294e+00],
                           [8.65522237e-04, 1.42051786e+00, -2.38450764e+02],
                           [3.00398868e-06, 5.60839152e-04, 1.00000000e+00]])
            down_stitch = Combine_Images(img_up, up_stitch, preCompH=dH)
            lH = np.array([[2.69049952e+00, -4.83562777e-01, -5.33252634e+02],
                           [1.36699921e+00, 1.79806045e+00, -3.41515826e+02],
                           [1.59202563e-03, -1.57472528e-04, 1.00000000e+00]])
            left_stitch = Combine_Images(img_left, down_stitch, preCompH=lH)
            rH = np.array([[-3.28865853e-01, 2.57128981e-02, 1.43058128e+03],
                           [-7.20883784e-01, 6.56432622e-01, 7.36877051e+02],
                           [-6.00176737e-04, -7.46517019e-05, 1.00000000e+00]])
            right_stitch = Combine_Images(img_right, left_stitch, preCompH=rH)
            resized_image = padded_resize(right_stitch, (640, 640))
            return resized_image
        except Exception as e:
            rospy.logwarn("Error in stitching images: %s", str(e))
            return None


def padded_resize(img, out_size):
    h, w = img.shape[:2]
    resized_w, resized_h = out_size
    scale = min(resized_w / w, resized_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top_pad = (resized_h - new_h) // 2
    bottom_pad = resized_h - new_h - top_pad
    left_pad = (resized_w - new_w) // 2
    right_pad = resized_w - new_w - left_pad
    padded_img = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img

def Combine_Images(img1, img2, draw=0, preCompH=None, loginfo=False):
    if preCompH is not None:
        H = preCompH
    else:
        keypoints1, descriptors1 = Create_Akaze(img1)
        keypoints2, descriptors2 = Create_Akaze(img2)
        kp_1_mat = keypoints_to_mat(keypoints1)
        kp_2_mat = keypoints_to_mat(keypoints2)
        desc_1_binarize = binarize(descriptors1).astype(np.float32)
        desc_2_binarize = binarize(descriptors2).astype(np.float32)
        match_pts = matches(desc_1_binarize, desc_2_binarize, 0.75)
        match_coords = get_point_coords(kp_1_mat, kp_2_mat, match_pts)
        H = RANSAC_homography(match_coords, eps=1, nIters=2000)
        if loginfo:
            rospy.loginfo(H)
    img2_c = np.array([[0, 0], [0, img2.shape[0]], [img2.shape[1], 0], [img2.shape[1], img2.shape[0]]])
    img1_c = np.array([[0, 0], [0, img1.shape[0]], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]]])
    img1_c_transformed = transform_homography(img1_c, H)
    img1_img2_corners = np.concatenate((img1_c_transformed, img2_c))
    min_x, min_y = np.int32(img1_img2_corners.min(axis=0))
    max_x, max_y = np.int32(img1_img2_corners.max(axis=0))
    trans_mat = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
    H_combined = np.dot(trans_mat, H)
    img1_warped = Warp_Image(img1, H_combined, (max_y - min_y, max_x - min_x))
    img2_warped = Warp_Image(img2, trans_mat, (max_y - min_y, max_x - min_x))
    stitched_img = np.where(img1_warped > 0, img1_warped, img2_warped)
    return stitched_img

def Warp_Image(img, H, out_shape):
    # H_inv = np.linalg.inv(H)
    # h, w = out_shape
    # x, y = np.meshgrid(np.arange(w), np.arange(h))
    # homogenous_pts = np.stack((x.ravel(), y.ravel()), axis=1)
    # transformed_pts = transform_homography(homogenous_pts, H_inv)
    # orig_x = transformed_pts[:, 0].reshape((h, w))
    # orig_y = transformed_pts[:, 1].reshape((h, w))
    # warped_image = np.zeros((h, w, img.shape[2]), dtype=np.uint8)
    # orig_x_clipped = np.clip(orig_x, 0, img.shape[1] - 1)
    # orig_y_clipped = np.clip(orig_y, 0, img.shape[0] - 1)
    # x_int = np.floor(orig_x_clipped).astype(np.int32)
    # y_int = np.floor(orig_y_clipped).astype(np.int32)
    # x_weight = orig_x_clipped - x_int
    # y_weight = orig_y_clipped - y_int
    # x1 = np.clip(x_int + 1, 0, img.shape[1] - 1)
    # y1 = np.clip(y_int + 1, 0, img.shape[0] - 1)
    # for c in range(img.shape[2]):
    #     top_left = (1 - x_weight) * (1 - y_weight) * img[y_int, x_int, c]
    #     top_right = x_weight * (1 - y_weight) * img[y_int, x1, c]
    #     bottom_left = (1 - x_weight) * y_weight * img[y1, x_int, c]
    #     bottom_right = x_weight * y_weight * img[y1, x1, c]
    #     warped_image[..., c] = top_left + top_right + bottom_left + bottom_right
    # out_of_bounds_mask = (orig_x < 0) | (orig_x >= img.shape[1]) | (orig_y < 0) | (orig_y >= img.shape[0])
    # warped_image[out_of_bounds_mask] = 0
    # return warped_image
    H_inv = np.linalg.inv(H)
    h, w = out_shape

    # Generate grid of (x, y) coordinates and create homogeneous coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    homogenous_pts = np.stack((x.ravel(), y.ravel(), np.ones_like(x).ravel()), axis=1)

    # Transform coordinates using the inverse homography
    transformed_pts = (H_inv @ homogenous_pts.T).T
    orig_x = (transformed_pts[:, 0] / transformed_pts[:, 2]).reshape(h, w)
    orig_y = (transformed_pts[:, 1] / transformed_pts[:, 2]).reshape(h, w)

    # Clip coordinates to image bounds
    orig_x_clipped = np.clip(orig_x, 0, img.shape[1] - 1)
    orig_y_clipped = np.clip(orig_y, 0, img.shape[0] - 1)

    # Compute integer and fractional parts of the coordinates
    x_int = orig_x_clipped.astype(np.int32)
    y_int = orig_y_clipped.astype(np.int32)
    x_frac = orig_x_clipped - x_int
    y_frac = orig_y_clipped - y_int

    x1 = np.clip(x_int + 1, 0, img.shape[1] - 1)
    y1 = np.clip(y_int + 1, 0, img.shape[0] - 1)

    warped_image = np.zeros((h, w, img.shape[2]), dtype=np.uint8)

    for c in range(img.shape[2]):
        top_left = img[y_int, x_int, c] * (1 - x_frac) * (1 - y_frac)
        top_right = img[y_int, x1, c] * x_frac * (1 - y_frac)
        bottom_left = img[y1, x_int, c] * (1 - x_frac) * y_frac
        bottom_right = img[y1, x1, c] * x_frac * y_frac

        warped_image[..., c] = top_left + top_right + bottom_left + bottom_right

    out_of_bounds_mask = (orig_x < 0) | (orig_x >= img.shape[1]) | (orig_y < 0) | (orig_y >= img.shape[0])
    warped_image[out_of_bounds_mask] = 0

    return warped_image


def Create_Akaze(img):
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(img, None)
    return keypoints, descriptors

def keypoints_to_mat(keypoint):
    kp = np.ones((len(keypoint), 4))
    for i in range(len(keypoint)):
        kp[i, 0:2] = keypoint[i].pt
        kp[i, 2] = keypoint[i].angle
        kp[i, 3] = keypoint[i].octave
    return kp

def binarize(descriptor):
    binary_output = np.ones((descriptor.shape[0], descriptor.shape[1] * 8))
    for i in range(descriptor.shape[0]):
        for j in range(descriptor.shape[1]):
            binary_output[i, j * 8:((j + 1) * 8)] = bits_map[descriptor[i, j]]
    return binary_output

def matches(desc1, desc2, ratioThreshhold):
    desc1_sq_sum = np.sum(desc1**2, axis=1, keepdims=True)
    desc2_sq_sum = np.sum(desc2**2, axis=1, keepdims=True)
    distance = desc1_sq_sum + desc2_sq_sum.T - np.dot((2 * desc1), desc2.T)
    distance_mat = np.sqrt(np.maximum(distance, 0))
    match_pts = []
    sorted_ind = np.argsort(distance_mat, axis=1)
    sorted_dist = np.take_along_axis(distance_mat, sorted_ind, axis=1)
    nearest_neightbor_ratio = sorted_dist[:, 0] / sorted_dist[:, 1]
    ratio_True = nearest_neightbor_ratio <= ratioThreshhold
    for i in range(len(ratio_True)):
        if ratio_True[i] == 1:
            match_pts.append((i, sorted_ind[i, 0]))
    return np.array(match_pts)

def get_point_coords(keypoint1, keypoint2, match_pts):
    return np.hstack((keypoint1[match_pts[:, 0], 0:2], keypoint2[match_pts[:, 1], 0:2]))

def RANSAC_homography(XY_Pairs, eps=1, nIters=100):
    bestH = np.ones((3, 3))
    bestInlierCount = -1
    bestInliers = np.zeros((XY_Pairs.shape[0],))
    for i in range(nIters):
        sample_ind = np.random.choice(XY_Pairs.shape[0], 4, replace=False)
        samples = XY_Pairs[sample_ind, :]
        H = homography_fitter(samples)
        test_fit = transform_homography(XY_Pairs[:, 0:2], H)
        loss_2 = np.sum((test_fit - XY_Pairs[:, 2:4])**2, axis=1, keepdims=True)
        loss = np.sqrt(loss_2)
        numInliers = np.sum(loss < eps)
        if numInliers > bestInlierCount:
            bestH = H
            bestInlierCount = numInliers
            bestInliers = XY_Pairs[(loss < eps).flatten()]
    finalH = homography_fitter(bestInliers)
    return finalH

def homography_fitter(XY_Pairs):
    A = np.zeros((2 * XY_Pairs.shape[0], 9))
    for i in range(A.shape[0]):
        if i % 2 == 0:
            A[i, 0] = XY_Pairs[i // 2, 0]
            A[i, 1] = XY_Pairs[i // 2, 1]
            A[i, 2] = 1
            A[i, 6] = -XY_Pairs[i // 2, 2] * XY_Pairs[i // 2, 0]
            A[i, 7] = -XY_Pairs[i // 2, 2] * XY_Pairs[i // 2, 1]
            A[i, 8] = -XY_Pairs[i // 2, 2]
        else:
            A[i, 3] = XY_Pairs[i // 2, 0]
            A[i, 4] = XY_Pairs[i // 2, 1]
            A[i, 5] = 1
            A[i, 6] = -XY_Pairs[i // 2, 3] * XY_Pairs[i // 2, 0]
            A[i, 7] = -XY_Pairs[i // 2, 3] * XY_Pairs[i // 2, 1]
            A[i, 8] = -XY_Pairs[i // 2, 3]
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T, A))
    H = eigenvectors[:, np.argmin(eigenvalues)].reshape((3, 3))
    return H / H[2, 2]

def transform_homography(XY_Coords, H):
    XY_Coords_homogenous = np.hstack((XY_Coords, np.ones((XY_Coords.shape[0], 1))))
    XY_Coords_H_Trasformed = np.dot(H, XY_Coords_homogenous.T).T
    return XY_Coords_H_Trasformed[:, :2] / XY_Coords_H_Trasformed[:, 2][:, None]


if __name__ == '__main__':
    for i in range(256):
        bits = bin(i)[2:].rjust(8, '0')
        bits_map.append(np.array([float(bit) for bit in bits]))

    rospy.init_node('image_stitcher')
    ImageStitcher()
    rospy.spin()

# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np

# bits_map=[]

# def padded_resize(img, out_size):
#     '''
#     Performs a padded resize on the input image in
#     order to maintain the same aspect ratio. The output 
#     image size is given by the input parameter
#     out_size.
#     '''
#     h, w = img.shape[:2]
#     resized_w, resized_h = out_size

#     # Calculates the largest possible scaling factor without falling out of bounds
#     scale = min(resized_w / w, resized_h / h)
#     new_w, new_h = int(w * scale), int(h * scale)

#     # Resize the image
#     resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

#     # Create a new image with the output size and fill with black on padded locations
#     top_pad = (resized_h - new_h) // 2
#     bottom_pad = resized_h - new_h - top_pad
#     left_pad = (resized_w - new_w) // 2
#     right_pad = resized_w - new_w - left_pad

#     padded_img = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

#     return padded_img

# def matches(desc1, desc2, ratioThreshhold):
#     '''
#     Given two descriptor vectors, this function returns an Nx2 matrix of 
#     '''
#     desc1_sq_sum = np.sum(desc1**2, axis=1, keepdims=True)
#     desc2_sq_sum = np.sum(desc2**2, axis=1, keepdims=True)
#     distance = desc1_sq_sum + desc2_sq_sum.T - np.dot((2*desc1), desc2.T)
#     distance_mat = np.sqrt(np.maximum(distance,0))
#     match_pts = []
#     sorted_ind = np.argsort(distance_mat, axis=1)
#     sorted_dist = np.take_along_axis(distance_mat, sorted_ind, axis=1)
#     nearest_neightbor_ratio = sorted_dist[:,0]/sorted_dist[:,1]
#     ratio_True = nearest_neightbor_ratio <= ratioThreshhold
#     for i in range (len(ratio_True)):
#         if ratio_True[i] == 1:
#             match_pts.append((i, sorted_ind[i,0]))
#     return np.array(match_pts)

# def draw_matches(img1, img2, kp1, kp2, match_pts):
#     '''
#     Creates an output image where the two source images stacked vertically
#     connecting matching keypoints with a line. 
        
#     Input - img1: Input image 1 of shape (H1,W1,3)
#             img2: Input image 2 of shape (H2,W2,3)
#             kp1: Keypoint matrix for image 1 of shape (N,4)
#             kp2: Keypoint matrix for image 2 of shape (M,4)
#             matches: List of matching pairs indices between the 2 sets of 
#                      keypoints (K,2)
    
#     Output - Image where 2 input images stacked vertically with lines joining 
#              the matched keypoints
#     Hint: see cv2.line
#     '''
#     #Hint:
#     #Use common.get_match_points() to extract keypoint locations
#     output = np.vstack((img1, img2))
#     for match in match_pts:
#         cv2.line(output, (int(match[0]),int(match[1])), (int(match[2]), int(match[3]+img1.shape[0])), (0,255,0))
#     return output

# def resize_image(image, scale_percent):
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
#     return resized

# def transform_homography(XY_Coords, H):
#     '''
#     Performs a homography transformation to a set of XY coordinates given a homography matrix H
    
#     Input: 
#         XY_Coords: Set of (x,y) coordinates (N,2) size matrix
#         H: (3,3) size homography matrix
#     Ouput:
#         Transformed XY coordinates (N,2) size matrix
#     '''
#     XY_Coords_homogenous = np.hstack((XY_Coords, np.ones((XY_Coords.shape[0],1))))
#     XY_Coords_H_Trasformed = np.dot(H, XY_Coords_homogenous.T).T
#     return XY_Coords_H_Trasformed[:,:2]/XY_Coords_H_Trasformed[:,2][:,None]

# def binarize(descriptor):
#     binary_output = np.ones((descriptor.shape[0], descriptor.shape[1]*8))

#     for i in range (descriptor.shape[0]):
#         for j in range (descriptor.shape[1]):
#             binary_output[i,j*8:((j+1)*8)] = bits_map[descriptor[i,j]]
#     return binary_output

# def keypoints_to_mat(keypoint):
#     kp = np.ones((len(keypoint), 4))
#     for i in range(len(keypoint)):
#         kp[i,0:2] = keypoint[i].pt
#         kp[i,2] = keypoint[i].angle
#         kp[i,3] = keypoint[i].octave
#     return kp

# def get_point_coords(keypoint1, keypoint2, match_pts):
#     return np.hstack((keypoint1[match_pts[:,0],0:2], keypoint2[match_pts[:,1],0:2]))
 
# def homography_fitter(XY_Pairs):
#     A = np.zeros((2*XY_Pairs.shape[0],9))
#     for i in range(A.shape[0]):
#         if i % 2 == 0:
#             A[i, 0] = XY_Pairs[i//2,0]
#             A[i, 1] = XY_Pairs[i//2,1]
#             A[i, 2] = 1
#             A[i, 6] = -XY_Pairs[i//2,2]*XY_Pairs[i//2,0]
#             A[i, 7] = -XY_Pairs[i//2,2]*XY_Pairs[i//2,1]
#             A[i, 8] = -XY_Pairs[i//2,2]
#         else:
#             A[i, 3] = XY_Pairs[i//2,0]
#             A[i, 4] = XY_Pairs[i//2,1]
#             A[i, 5] = 1
#             A[i, 6] = -XY_Pairs[i//2,3]*XY_Pairs[i//2,0]
#             A[i, 7] = -XY_Pairs[i//2,3]*XY_Pairs[i//2,1]
#             A[i, 8] = -XY_Pairs[i//2,3]
#     eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T, A))
#     H = eigenvectors[:, np.argmin(eigenvalues)].reshape((3,3))
#     return H/H[2,2]

# def RANSAC_homography(XY_Pairs, eps=1, nIters=100):
#     '''
#     Performs RANSAC algorithm to find a homography which best fits XY correspondences

#     Input:

#     Output:
#     '''
#     bestH = np.ones((3,3))
#     bestInlierCount = -1
#     bestInliers = np.zeros((XY_Pairs.shape[0],))
#     for i in range(nIters):
#         sample_ind = np.random.choice(XY_Pairs.shape[0], 4, replace=False)
#         samples = XY_Pairs[sample_ind, :]
#         H = homography_fitter(samples)
#         test_fit = transform_homography(XY_Pairs[:,0:2], H)
#         loss_2 = np.sum((test_fit - XY_Pairs[:, 2:4])**2, axis=1, keepdims=True)
#         loss = np.sqrt(loss_2)
#         numInliers = np.sum(loss<eps)
#         if numInliers > bestInlierCount:
#             bestH = H
#             bestInlierCount = numInliers
#             bestInliers = XY_Pairs[(loss<eps).flatten()]
#     finalH = homography_fitter(bestInliers)
#     return finalH

# def Create_Akaze(img):
#     akaze = cv2.AKAZE_create()
#     keypoints, descriptors = akaze.detectAndCompute(img, None)
#     return keypoints, descriptors
    
# # def Warp_Image(img, H, out_shape):
# #     H_inv = np.linalg.inv(H)
# #     h, w = out_shape
# #     warped_image = np.zeros((h, w, 3), dtype=np.uint8)
# #     for y in range(h):
# #         for x in range(w):
# #             homogenous_pt = np.array([[x, y]])
# #             orig_coord = transform_homography(homogenous_pt, H_inv)
# #             orig_x = orig_coord[0,0]
# #             orig_y = orig_coord[0,1]
# #             if 0 <= orig_x <img.shape[1] and 0 <= orig_y < img.shape[0]:
# #                 x_int, y_int = int(orig_x), int(orig_y)
# #                 x_1, y_1 = min(x_int + 1, img.shape[1] - 1), min(y_int + 1, img.shape[0] - 1)
# #                 weight_x = orig_x - x_int
# #                 weight_y = orig_y - y_int
# #                 tlw = (1 - weight_x) * (1 - weight_y) 
# #                 trw = weight_x * (1 - weight_y)
# #                 blw = (1 - weight_x) * weight_y
# #                 brw =  weight_x * weight_y 
# #                 for channel in range(img.shape[2]):
# #                     top_left = tlw * img[y_int, x_int, channel]
# #                     top_right = trw * img[y_int, x_1, channel]
# #                     bottom_left = blw * img[y_1, x_int, channel]
# #                     bottom_right = brw * img[y_1, x_int, channel]
# #                     warped_image[y, x, channel] = top_left + top_right + bottom_left + bottom_right
    
# #     return warped_image


# def Warp_Image(img, H, out_shape):
#     H_inv = np.linalg.inv(H)
#     h, w = out_shape

#     # Generate grid of (x, y) coordinates
#     x, y = np.meshgrid(np.arange(w), np.arange(h))

#     # Flatten and create homogeneous coordinates
#     homogenous_pts = np.stack((x.ravel(), y.ravel()), axis=1)

#     # Transform coordinates using the inverse homography
#     transformed_pts = transform_homography(homogenous_pts, H_inv)

#     # Extract and reshape coordinates
#     orig_x = transformed_pts[:, 0].reshape((h, w))
#     orig_y = transformed_pts[:, 1].reshape((h, w))

#     # Initialize warped image
#     warped_image = np.zeros((h, w, img.shape[2]), dtype=np.uint8)

#     # Clip coordinates to image bounds
#     orig_x_clipped = np.clip(orig_x, 0, img.shape[1] - 1)
#     orig_y_clipped = np.clip(orig_y, 0, img.shape[0] - 1)

#     x_int = np.floor(orig_x_clipped).astype(np.int32)
#     y_int = np.floor(orig_y_clipped).astype(np.int32)

#     x_weight = orig_x_clipped - x_int
#     y_weight = orig_y_clipped - y_int

#     x1 = np.clip(x_int + 1, 0, img.shape[1] - 1)
#     y1 = np.clip(y_int + 1, 0, img.shape[0] - 1)
#     for c in range(img.shape[2]):
#         top_left = (1 - x_weight) * (1 - y_weight) * img[y_int, x_int, c]
#         top_right = x_weight * (1 - y_weight) * img[y_int, x1, c]
#         bottom_left = (1 - x_weight) * y_weight * img[y1, x_int, c]
#         bottom_right = x_weight * y_weight * img[y1, x1, c]

#         warped_image[..., c] = top_left + top_right + bottom_left + bottom_right

#     # Set out-of-bounds areas to black
#     out_of_bounds_mask = (orig_x < 0) | (orig_x >= img.shape[1]) | (orig_y < 0) | (orig_y >= img.shape[0])
#     warped_image[out_of_bounds_mask] = 0

#     return warped_image


# def Combine_Images(img1, img2, draw=0, preCompH=None, loginfo=False):
#     H = []
#     if preCompH is not None:
#         H = preCompH
#     else:
#         keypoints1, descriptors1 = Create_Akaze(img1)
#         keypoints2, descriptors2 = Create_Akaze(img2)
#         kp_1_mat = keypoints_to_mat(keypoints1)
#         kp_2_mat = keypoints_to_mat(keypoints2)
#         desc_1_binarize = binarize(descriptors1).astype(np.float32)
#         desc_2_binarize = binarize(descriptors2).astype(np.float32)
#         ratioThreshhold = 0.75
#         match_pts = matches(desc_1_binarize, desc_2_binarize, ratioThreshhold)
#         match_coords = get_point_coords(kp_1_mat, kp_2_mat, match_pts)
#         # if draw == 1:
#         #     out_1 = cv2.drawKeypoints(img1, keypoints1, None, color=(0, 0, 255))
#         #     out_2 = cv2.drawKeypoints(img2, keypoints2, None, color=(0, 0, 255))
#         #     cv2.imwrite('desc1.jpg', out_1)
#         #     cv2.imwrite('desc2.jpg', out_2)
#         #     debugging = draw_matches(out_1, out_2, kp_1_mat, kp_2_mat, match_coords)
#         #     cv2.imwrite('all_matches.jpg', debugging)
#         H = RANSAC_homography(match_coords, eps=1, nIters=2000)
#         if loginfo==True:
#             rospy.loginfo(H)
#     img2_c = np.array([[0, 0], [0, img2.shape[0]], [img2.shape[1], 0], [img2.shape[1], img2.shape[0]]])
#     img1_c = np.array([[0, 0], [0, img1.shape[0]], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]]])
#     img1_c_transformed = transform_homography(img1_c, H)
#     img1_img2_corners = np.concatenate((img1_c_transformed, img2_c))
#     min_x, min_y = np.int32(img1_img2_corners.min(axis=0))
#     max_x, max_y = np.int32(img1_img2_corners.max(axis=0))
#     trans_mat = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
#     H_combined = np.dot(trans_mat, H)
#     img1_warped = Warp_Image(img1, H_combined, (max_y-min_y, max_x-min_x))
#     img2_warped = Warp_Image(img2, trans_mat, (max_y-min_y, max_x-min_x))
#     stitched_img = np.where(img1_warped > 0, img1_warped, img2_warped)
#     return stitched_img



# class ImageStitcher:
#     def __init__(self):
#         self.bridge = CvBridge()
#         self.left_image = None
#         self.front_image = None
#         self.right_image = None
#         self.up_image = None
#         self.down_image = None

#         rospy.Subscriber('webcam_left_front/image_raw_left_front', Image, self.left_callback)
#         rospy.Subscriber('webcam_front/image_raw_front', Image, self.front_callback)
#         rospy.Subscriber('webcam_right_front/image_raw_right_front', Image, self.right_callback)
#         rospy.Subscriber('webcam_up/image_raw_up', Image, self.up_callback)
#         rospy.Subscriber('webcam_down/image_raw_down', Image, self.down_callback)
        
#         self.pub = rospy.Publisher('stitched_image', Image, queue_size=1)

#     def left_callback(self, msg):
#         self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         self.stitch_and_publish()

#     def front_callback(self, msg):
#         self.front_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         self.stitch_and_publish()

#     def right_callback(self, msg):
#         self.right_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         self.stitch_and_publish()
    
#     def down_callback(self, msg):
#         self.down_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         self.stitch_and_publish()

#     def up_callback(self, msg):
#         self.up_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         self.stitch_and_publish()

#     def stitch_and_publish(self):
#         if self.left_image is not None and self.front_image is not None and self.right_image is not None and self.down_image is not None and self.up_image is not None:
#             img_left = self.left_image#cv2.cvtColor(self.left_image, cv2.COLOR_BGR2RGB)
#             img_front = self.front_image#cv2.cvtColor(self.front_image, cv2.COLOR_BGR2RGB)
#             img_right = self.right_image#cv2.cvtColor(self.right_image, cv2.COLOR_BGR2RGB)
#             img_up = self.up_image#cv2.cvtColor(self.up_image, cv2.COLOR_BGR2RGB)
#             img_down = self.down_image#cv2.cvtColor(self.down_image, cv2.COLOR_BGR2RGB)
#             #stitched_img = Combine_Images(img_left, img_front)
#             #stitched_img_left = Combine_Images(img_left1, img_left)
#             #stitched_img_right = Combine_Images(img_right, img_front)
#             #final_stitch = Combine_Images(stitched_img_left, stitched_img_right)
#             #cv2.imwrite('stitch.jpg', final_stitch)
#             scale_percent = 70
#             #stitched_imgR = resize_image(stitched_img, scale_percent)
#             #cv2.imshow('l', img_left)
#             #cv2.imshow('f', img_front)
#             #cv2.imshow('l', self.left_image)
#             #final_stitched_img = Combine_Images(img_right, stitched_img)
#             #final_stitched_imgR = resize_image(final_stitched_img, scale_percent)
#             #img_left1R = resize_image(img_left1, 200)
#             #final_final_img = Combine_Images(final_stitched_img, img_left1)
#             #rospy.loginfo(final_final_img.shape)
#             #up_stitch = Combine_Images(img_up, final_stitched_img)
#             #up_stitch = resize_image(up_stitch, scale_percent)
#             #down_stitch = Combine_Images(img_down, up_stitch)
#             upH = np.array([[ 8.77117773e-01,-1.96206979e-01,6.11303304e+01],
#                             [-5.99665375e-04,7.04209450e-01,1.68395999e+02],
#                             [-9.20787802e-07,-3.92825216e-04,1.00000000e+00]])
#             up_stitch = Combine_Images(img_down, img_front, 1, preCompH=upH)
#             #cv2.imwrite('used_forposter.jpg', up_stitch)
#             dH = np.array([[1.25065679e+00,3.49143801e-01,-3.98029294e+00],
#                            [8.65522237e-04,1.42051786e+00,-2.38450764e+02],
#                            [3.00398868e-06,5.60839152e-04,1.00000000e+00]])
#             down_stitch = Combine_Images(img_up, up_stitch, preCompH=dH)
#             lH = np.array([[2.69049952e+00,-4.83562777e-01,-5.33252634e+02],
#                            [1.36699921e+00,1.79806045e+00,-3.41515826e+02],
#                            [1.59202563e-03,-1.57472528e-04,1.00000000e+00]])
#             left_stitch = Combine_Images(img_left, down_stitch, preCompH=lH)
#             rH = np.array([[-3.28865853e-01,2.57128981e-02,1.43058128e+03],
#                            [-7.20883784e-01,6.56432622e-01,7.36877051e+02],
#                            [-6.00176737e-04,-7.46517019e-05,1.00000000e+00]])
#             right_stitch = Combine_Images(img_right, left_stitch, preCompH=rH)
#             resized_image = padded_resize(right_stitch, (640, 640))
#             cv2.imwrite('resized_image.jpg', resized_image)
#             #cv2.imwrite('stitch1.jpg', img_up)
#             #cv2.imwrite('stitch2.jpg', img_down)
#             stitched_msg = self.bridge.cv2_to_imgmsg(resized_image, 'bgr8')
#             self.pub.publish(stitched_msg)
#             #cv2.imwrite('stitch2.jpg', img_left)
#             #cv2.imwrite('stitch2.jpg', img_left1)



# if __name__ == '__main__':
#     for i in range(256):
#         bits = bin(i)[2:].rjust(8,'0')
#         bits_map.append(np.array([float(bit) for bit in bits]))

#     rospy.init_node('image_stitcher')
#     ImageStitcher()
#     rospy.spin()
# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# import threading
# import message_filters

# bits_map = []

# class ImageStitcher:
#     def __init__(self):
#         self.bridge = CvBridge()
#         self.images = {
#             'left': None,
#             'front': None,
#             'right': None,
#             'up': None,
#             'down': None
#         }
#         self.lock = threading.Lock()

#         left_sub = message_filters.Subscriber('webcam_left_front/image_raw_left_front', Image)
#         front_sub = message_filters.Subscriber('webcam_front/image_raw_front', Image)
#         right_sub = message_filters.Subscriber('webcam_right_front/image_raw_right_front', Image)
#         up_sub = message_filters.Subscriber('webcam_up/image_raw_up', Image)
#         down_sub = message_filters.Subscriber('webcam_down/image_raw_down', Image)

#         ts = message_filters.ApproximateTimeSynchronizer(
#             [left_sub, front_sub, right_sub, up_sub, down_sub], queue_size=10, slop=0.01
#         )
#         ts.registerCallback(self.image_callback)

#         self.pub = rospy.Publisher('stitched_image', Image, queue_size=1)
#         self.stitching_thread = threading.Thread(target=self.stitch_and_publish)
#         self.stitching_thread.daemon = True
#         self.stitching_thread.start()

#     def image_callback(self, left, front, right, up, down):
#         with self.lock:
#             self.images['left'] = self.bridge.imgmsg_to_cv2(left, "bgr8")
#             self.images['front'] = self.bridge.imgmsg_to_cv2(front, "bgr8")
#             self.images['right'] = self.bridge.imgmsg_to_cv2(right, "bgr8")
#             self.images['up'] = self.bridge.imgmsg_to_cv2(up, "bgr8")
#             self.images['down'] = self.bridge.imgmsg_to_cv2(down, "bgr8")

#     def stitch_and_publish(self):
#         rate = rospy.Rate(20)  # Adjust the rate as needed
#         while not rospy.is_shutdown():
#             with self.lock:
#                 if all(image is not None for image in self.images.values()):
#                     stitched_image = self.stitch_images()
#                     if stitched_image is not None:
#                         stitched_msg = self.bridge.cv2_to_imgmsg(stitched_image, 'bgr8')
#                         self.pub.publish(stitched_msg)
#                         # Reset images to avoid re-stitching the same set
#                         self.images = {key: None for key in self.images}
#             rate.sleep()

#     def stitch_images(self):
#         img_left = self.images['left']
#         img_front = self.images['front']
#         img_right = self.images['right']
#         img_up = self.images['up']
#         img_down = self.images['down']

#         try:
#             upH = np.array([[ 8.77117773e-01, -1.96206979e-01, 6.11303304e+01],
#                             [-5.99665375e-04, 7.04209450e-01, 1.68395999e+02],
#                             [-9.20787802e-07, -3.92825216e-04, 1.00000000e+00]])
#             up_stitch = Combine_Images(img_down, img_front, preCompH=upH)
#             dH = np.array([[1.25065679e+00, 3.49143801e-01, -3.98029294e+00],
#                            [8.65522237e-04, 1.42051786e+00, -2.38450764e+02],
#                            [3.00398868e-06, 5.60839152e-04, 1.00000000e+00]])
#             down_stitch = Combine_Images(img_up, up_stitch, preCompH=dH)
#             lH = np.array([[2.69049952e+00, -4.83562777e-01, -5.33252634e+02],
#                            [1.36699921e+00, 1.79806045e+00, -3.41515826e+02],
#                            [1.59202563e-03, -1.57472528e-04, 1.00000000e+00]])
#             left_stitch = Combine_Images(img_left, down_stitch, preCompH=lH)
#             rH = np.array([[-3.28865853e-01, 2.57128981e-02, 1.43058128e+03],
#                            [-7.20883784e-01, 6.56432622e-01, 7.36877051e+02],
#                            [-6.00176737e-04, -7.46517019e-05, 1.00000000e+00]])
#             right_stitch = Combine_Images(img_right, left_stitch, preCompH=rH)
#             resized_image = padded_resize(right_stitch, (640, 640))
#             return resized_image
#         except Exception as e:
#             rospy.logwarn("Error in stitching images: %s", str(e))
#             return None


# def padded_resize(img, out_size):
#     h, w = img.shape[:2]
#     resized_w, resized_h = out_size
#     scale = min(resized_w / w, resized_h / h)
#     new_w, new_h = int(w * scale), int(h * scale)
#     resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#     top_pad = (resized_h - new_h) // 2
#     bottom_pad = resized_h - new_h - top_pad
#     left_pad = (resized_w - new_w) // 2
#     right_pad = resized_w - new_w - left_pad
#     padded_img = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#     return padded_img

# def Combine_Images(img1, img2, draw=0, preCompH=None, loginfo=False):
#     if preCompH is not None:
#         H = preCompH
#     else:
#         keypoints1, descriptors1 = Create_Akaze(img1)
#         keypoints2, descriptors2 = Create_Akaze(img2)
#         kp_1_mat = keypoints_to_mat(keypoints1)
#         kp_2_mat = keypoints_to_mat(keypoints2)
#         desc_1_binarize = binarize(descriptors1).astype(np.float32)
#         desc_2_binarize = binarize(descriptors2).astype(np.float32)
#         match_pts = matches(desc_1_binarize, desc_2_binarize, 0.75)
#         match_coords = get_point_coords(kp_1_mat, kp_2_mat, match_pts)
#         H = RANSAC_homography(match_coords, eps=1, nIters=2000)
#         if loginfo:
#             rospy.loginfo(H)
#     img2_c = np.array([[0, 0], [0, img2.shape[0]], [img2.shape[1], 0], [img2.shape[1], img2.shape[0]]])
#     img1_c = np.array([[0, 0], [0, img1.shape[0]], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]]])
#     img1_c_transformed = transform_homography(img1_c, H)
#     img1_img2_corners = np.concatenate((img1_c_transformed, img2_c))
#     min_x, min_y = np.int32(img1_img2_corners.min(axis=0))
#     max_x, max_y = np.int32(img1_img2_corners.max(axis=0))
#     trans_mat = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
#     H_combined = np.dot(trans_mat, H)
#     img1_warped = Warp_Image(img1, H_combined, (max_y - min_y, max_x - min_x))
#     img2_warped = Warp_Image(img2, trans_mat, (max_y - min_y, max_x - min_x))
#     stitched_img = np.where(img1_warped > 0, img1_warped, img2_warped)
#     return stitched_img

# def Warp_Image(img, H, out_shape):
#     H_inv = np.linalg.inv(H)
#     h, w = out_shape
#     x, y = np.meshgrid(np.arange(w), np.arange(h))
#     homogenous_pts = np.stack((x.ravel(), y.ravel()), axis=1)
#     transformed_pts = transform_homography(homogenous_pts, H_inv)
#     orig_x = transformed_pts[:, 0].reshape((h, w))
#     orig_y = transformed_pts[:, 1].reshape((h, w))
#     warped_image = np.zeros((h, w, img.shape[2]), dtype=np.uint8)
#     orig_x_clipped = np.clip(orig_x, 0, img.shape[1] - 1)
#     orig_y_clipped = np.clip(orig_y, 0, img.shape[0] - 1)
#     x_int = np.floor(orig_x_clipped).astype(np.int32)
#     y_int = np.floor(orig_y_clipped).astype(np.int32)
#     x_weight = orig_x_clipped - x_int
#     y_weight = orig_y_clipped - y_int
#     x1 = np.clip(x_int + 1, 0, img.shape[1] - 1)
#     y1 = np.clip(y_int + 1, 0, img.shape[0] - 1)
#     for c in range(img.shape[2]):
#         top_left = (1 - x_weight) * (1 - y_weight) * img[y_int, x_int, c]
#         top_right = x_weight * (1 - y_weight) * img[y_int, x1, c]
#         bottom_left = (1 - x_weight) * y_weight * img[y1, x_int, c]
#         bottom_right = x_weight * y_weight * img[y1, x1, c]
#         warped_image[..., c] = top_left + top_right + bottom_left + bottom_right
#     out_of_bounds_mask = (orig_x < 0) | (orig_x >= img.shape[1]) | (orig_y < 0) | (orig_y >= img.shape[0])
#     warped_image[out_of_bounds_mask] = 0
#     return warped_image

# def Create_Akaze(img):
#     akaze = cv2.AKAZE_create()
#     keypoints, descriptors = akaze.detectAndCompute(img, None)
#     return keypoints, descriptors

# def keypoints_to_mat(keypoint):
#     kp = np.ones((len(keypoint), 4))
#     for i in range(len(keypoint)):
#         kp[i, 0:2] = keypoint[i].pt
#         kp[i, 2] = keypoint[i].angle
#         kp[i, 3] = keypoint[i].octave
#     return kp

# def binarize(descriptor):
#     binary_output = np.ones((descriptor.shape[0], descriptor.shape[1] * 8))
#     for i in range(descriptor.shape[0]):
#         for j in range(descriptor.shape[1]):
#             binary_output[i, j * 8:((j + 1) * 8)] = bits_map[descriptor[i, j]]
#     return binary_output

# def matches(desc1, desc2, ratioThreshhold):
#     desc1_sq_sum = np.sum(desc1**2, axis=1, keepdims=True)
#     desc2_sq_sum = np.sum(desc2**2, axis=1, keepdims=True)
#     distance = desc1_sq_sum + desc2_sq_sum.T - np.dot((2 * desc1), desc2.T)
#     distance_mat = np.sqrt(np.maximum(distance, 0))
#     match_pts = []
#     sorted_ind = np.argsort(distance_mat, axis=1)
#     sorted_dist = np.take_along_axis(distance_mat, sorted_ind, axis=1)
#     nearest_neightbor_ratio = sorted_dist[:, 0] / sorted_dist[:, 1]
#     ratio_True = nearest_neightbor_ratio <= ratioThreshhold
#     for i in range(len(ratio_True)):
#         if ratio_True[i] == 1:
#             match_pts.append((i, sorted_ind[i, 0]))
#     return np.array(match_pts)

# def get_point_coords(keypoint1, keypoint2, match_pts):
#     return np.hstack((keypoint1[match_pts[:, 0], 0:2], keypoint2[match_pts[:, 1], 0:2]))

# def RANSAC_homography(XY_Pairs, eps=1, nIters=100):
#     bestH = np.ones((3, 3))
#     bestInlierCount = -1
#     bestInliers = np.zeros((XY_Pairs.shape[0],))
#     for i in range(nIters):
#         sample_ind = np.random.choice(XY_Pairs.shape[0], 4, replace=False)
#         samples = XY_Pairs[sample_ind, :]
#         H = homography_fitter(samples)
#         test_fit = transform_homography(XY_Pairs[:, 0:2], H)
#         loss_2 = np.sum((test_fit - XY_Pairs[:, 2:4])**2, axis=1, keepdims=True)
#         loss = np.sqrt(loss_2)
#         numInliers = np.sum(loss < eps)
#         if numInliers > bestInlierCount:
#             bestH = H
#             bestInlierCount = numInliers
#             bestInliers = XY_Pairs[(loss < eps).flatten()]
#     finalH = homography_fitter(bestInliers)
#     return finalH

# def homography_fitter(XY_Pairs):
#     A = np.zeros((2 * XY_Pairs.shape[0], 9))
#     for i in range(A.shape[0]):
#         if i % 2 == 0:
#             A[i, 0] = XY_Pairs[i // 2, 0]
#             A[i, 1] = XY_Pairs[i // 2, 1]
#             A[i, 2] = 1
#             A[i, 6] = -XY_Pairs[i // 2, 2] * XY_Pairs[i // 2, 0]
#             A[i, 7] = -XY_Pairs[i // 2, 2] * XY_Pairs[i // 2, 1]
#             A[i, 8] = -XY_Pairs[i // 2, 2]
#         else:
#             A[i, 3] = XY_Pairs[i // 2, 0]
#             A[i, 4] = XY_Pairs[i // 2, 1]
#             A[i, 5] = 1
#             A[i, 6] = -XY_Pairs[i // 2, 3] * XY_Pairs[i // 2, 0]
#             A[i, 7] = -XY_Pairs[i // 2, 3] * XY_Pairs[i // 2, 1]
#             A[i, 8] = -XY_Pairs[i // 2, 3]
#     eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T, A))
#     H = eigenvectors[:, np.argmin(eigenvalues)].reshape((3, 3))
#     return H / H[2, 2]

# def transform_homography(XY_Coords, H):
#     XY_Coords_homogenous = np.hstack((XY_Coords, np.ones((XY_Coords.shape[0], 1))))
#     XY_Coords_H_Trasformed = np.dot(H, XY_Coords_homogenous.T).T
#     return XY_Coords_H_Trasformed[:, :2] / XY_Coords_H_Trasformed[:, 2][:, None]


# if __name__ == '__main__':
#     for i in range(256):
#         bits = bin(i)[2:].rjust(8, '0')
#         bits_map.append(np.array([float(bit) for bit in bits]))

#     rospy.init_node('image_stitcher')
#     ImageStitcher()
#     rospy.spin()