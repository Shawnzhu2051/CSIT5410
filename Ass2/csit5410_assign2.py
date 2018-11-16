import numpy as np
import cv2
import math

patch = 30
target_radius = 114
threshold = 137
FILENAME = 'qiqiu.png'


def put_point_in_circle_space(x, y, circle_area_data, width, height):
    if x < height-1 and y < width-1:
        circle_area_data[x, y] += 1
    return circle_area_data


def accumulate(x0, y0, radius, circle_area_data, width, height):
    x = radius
    y = 0
    flag = 1 - x
    while x >= y:
        circle_area_data = put_point_in_circle_space(x + x0, y + y0, circle_area_data, width, height)
        circle_area_data = put_point_in_circle_space(y + x0, x + y0, circle_area_data, width, height)
        circle_area_data = put_point_in_circle_space(-x + x0, y + y0, circle_area_data, width, height)
        circle_area_data = put_point_in_circle_space(-y + x0, x + y0, circle_area_data, width, height)
        circle_area_data = put_point_in_circle_space(-x + x0, -y + y0, circle_area_data, width, height)
        circle_area_data = put_point_in_circle_space(-y + x0, -x + y0, circle_area_data, width, height)
        circle_area_data = put_point_in_circle_space(x + x0, -y + y0, circle_area_data, width, height)
        circle_area_data = put_point_in_circle_space(y + x0, -x + y0, circle_area_data, width, height)
        y+=1
        if flag <= 0:
            flag += 2 * y + 1
        else:
            x -= 1
            flag += 2 * (y - x) + 1
    return circle_area_data


def get_edge_locations(edged_image):
    edges = np.where(edged_image == 255)
    return edges


def compute_circle_space_array(edges, width, height, target_radius):
    acc_array = np.zeros((height,width,target_radius+1))
    for i in range(0, len(edges[0])):
        x = edges[0][i]
        y = edges[1][i]
        acc_array[:, :, target_radius] = accumulate(x, y, target_radius, acc_array[:, :, target_radius], width, height)
    return acc_array


def plot_circle(output, acc_array, target_radius, width, height, threshold, patch):
    i, j = 0, 0
    center_loc_filter = np.ones((patch,patch, target_radius+1))
    while i < height-patch:
        while j < width-patch:
            center_loc_filter = acc_array[i: i + patch, j: j + patch, :] * center_loc_filter
            max_pt = np.where(center_loc_filter == center_loc_filter.max())
            x0 = max_pt[0]
            y0 = max_pt[1]
            radius = max_pt[2]
            y0 += j
            x0 += i
            if center_loc_filter.max() > threshold:
                if len(x0) > 1 and len(y0) > 1 and len(radius) > 1:
                    x0 = x0[0]
                    y0 = y0[0]
                    radius = radius[0]
                if x0 > y0:
                    cv2.circle(output, (x0, y0), radius, (255, 255, 255), 5)
                else:
                    cv2.circle(output, (y0, x0), radius, (255, 255, 255), 5)
            j = j + patch
            center_loc_filter[:, :, :] = 1
        j = 0
        i = i + patch
    return output


def task1():
    original = cv2.imread(FILENAME, 1)

    gray_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    edged_image = cv2.Canny(gray_image, 75, 150)

    height, width = gray_image.shape
    output = np.zeros((height, width, 3), np.uint8)

    edges = get_edge_locations(edged_image)

    acc_array = compute_circle_space_array(edges, width, height, target_radius)
    output = plot_circle(output, acc_array, target_radius, width, height, threshold, patch)
    cv2.imwrite('output.jpg', output)


def task2():
    X1 = np.array([[1,2], [2, 3], [3, 3], [4, 5], [5, 5]])
    X2 = np.array([[1, 0], [2, 1], [3, 1], [3, 2], [5, 3], [6, 5]])
    test = [2,5]
    mean_x1 = 0
    mean_y1 = 0
    mean_x2 = 0
    mean_y2 = 0
    for pair in X1:
        mean_x1 += pair[0]
        mean_y1 += pair[1]
    Mu1 = np.array([mean_x1 / 5, mean_y1 / 5])

    for pair in X2:
        mean_x2 += pair[0]
        mean_y2 += pair[1]
    Mu2 = np.array([mean_x2 / 6, mean_y2 / 6])
    S1 = np.cov(X1.T)
    S2 = np.cov(X2.T)
    Sw = S1 + S2
    SB = sum((Mu1 - Mu2) * ((Mu1 - Mu2).T))
    invSw = np.linalg.inv(Sw)
    inv_Sw_by_SB = invSw * SB
    D, V = np.linalg.eig(inv_Sw_by_SB)
    W = V[0]
    w_t = [[W[0]],[W[1]]]
    sp = 1 / 2 * np.array(w_t) * np.array(Mu1 + Mu2)
    k = (sp[0][1] - sp[1][1]) / (sp[0][0] - sp[1][0])
    a = sp[1][1] - (k * sp[1][0])
    if test[0] * k + a - test[1] < 0:
        print ('The predicted class number is: Class 1')
    else:
        print ('The predicted class number is: Class 2')
    print('The within-class scatter matrix is:')
    print(Sw)
    print('The weight vector is:')
    print(W)

if __name__ == "__main__":
    task1()
    task2()
