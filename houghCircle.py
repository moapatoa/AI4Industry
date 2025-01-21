import cv2
import numpy as np

blur_ratio = 21

def find2MainCircle_fromPath(image_path: str, blur_ratio: int =21,* , verbose: bool = False):
    # Load an image
    baseimage = cv2.imread(image_path, cv2.IMREAD_COLOR)

    return find2MainCircle(baseimage, blur_ratio, verbose=verbose)

def find2MainCircle(image_mat: np.ndarray, blur_ratio=21, *, verbose: bool = False):

    image = image_mat.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    gray = cv2.GaussianBlur(gray,(blur_ratio, blur_ratio) ,2)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                           param1=50, param2=30, minRadius=0, maxRadius=0)

    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

    if verbose:
        # Display the result
        cv2.imshow('Detected Circles', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # find clusters of Circles

    def find_clusters(circles, min_dist=10):
        circles = np.uint16(np.around(circles))
        clusters = []
        for i in circles[0, :]:
            if not clusters:
                clusters.append([i])
            else:
                for cluster in clusters:
                    for j in cluster:
                        if np.linalg.norm(np.array(i[:2]) - np.array(j[:2])) < min_dist:
                            cluster.append(i)
                            break
                    else:
                        continue
                    break
                else:
                    clusters.append([i])
        return clusters

    clusters = find_clusters(circles, min_dist=10)
    img_width = image.shape[1]

    left_circle = None
    for cluster in clusters:
        for i in cluster:
            if i[0] < img_width / 2:
                left_circle = i
                print(left_circle)
                break
        if left_circle is not None:
            break

    right_circle = None
    for cluster in clusters:
        for i in cluster:
            if i[0] >= img_width / 2:
                right_circle = i
                print(i)
                break
        if right_circle is not None:
            break

    output_circle = [left_circle, right_circle]

    # Draw clusters
    image2 = image_mat.copy()
    for i in output_circle:
        if i is None:
            continue
        # inner circle
        cv2.circle(image2, (i[0], i[1]), i[2] - 50, (255, 0, 0), 2)
        # outer circles
        cv2.circle(image2, (i[0], i[1]), i[2], (0,0, 255), 2)
        # center of the circle
        cv2.circle(image2, (i[0], i[1]), 2, (0, 0, 255), 3)

    if verbose:
        cv2.imshow('Detected Circles', image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    find2MainCircle_fromPath('./Images/frame_P1_24H_03.png', blur_ratio, verbose=True)
