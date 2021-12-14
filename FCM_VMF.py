import cv2
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
from fastfcm import FastFcm
import fitting

class FCMVMF():

    def __init__(self, image, n_clusters, m, max_iter, epsilon,
                 fcm_result, per_input, filter_size, gener_size, alpha):
        if np.ndim(image) != 2:
            raise Exception("<image> needs to be 2D (gray scale image).")
        if n_clusters <= 0 or n_clusters != int(n_clusters):
            raise Exception("<n_clusters> needs to be positive integer.")
        if m < 1:
            raise Exception("<m> needs to be >= 1.")

        self.image = image
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.per_input = per_input
        self.filter_size = filter_size
        self.gener_size = gener_size
        self.paddle_size = (self.gener_size - self.filter_size)/2


        self.shape = image.shape # image shape
        self.X = image.flatten().astype('float') # flatted image shape: (number of pixels,1)
        self.refer_x = np.zeros(self.X.shape)
        self.numPixels = image.size
        self.refer_img = np.zeros(self.shape)
        self.diff2 = np.zeros(self.shape)
        self.delta = np.zeros(self.shape)
        self.lamda = np.zeros(self.shape)
        self.eWeight = np.zeros(self.shape)
        self.filter_img = np.zeros(self.shape)
        self.check = 0
        self.fcm_result = fcm_result
        self.alpha = alpha

    def calc_permax(self, x_b, x_e, y_b, y_e):
        counter = np.zeros(self.n_clusters)
        for i in range(x_b, x_e):
            for j in range(y_b, y_e):
                counter[int(self.fcm_result[i, j])] = \
                    counter[int(self.fcm_result[i, j])] + 1

        total = np.sum(counter)
        max = np.max(counter)

        return max / total

    def calc_begin(self, x_b, x_e, y_b, y_e, d, gener_size):
        if x_b - d < 0:
            gener_x_b = 0
            filter_x_b = 0
        elif x_e + d >= self.image.shape[0]:
            gener_x_b = self.image.shape[0] - gener_size
            filter_x_b = d * 2
        else:
            gener_x_b = x_b - d
            filter_x_b = d

        if y_b - d < 0:
            gener_y_b = 0
            filter_y_b = 0
        elif y_e + d >= self.image.shape[1]:
            gener_y_b = self.image.shape[1] - gener_size
            filter_y_b = d * 2
        else:
            gener_y_b = y_b - d
            filter_y_b = d

        return gener_x_b, filter_x_b, gener_y_b, filter_y_b

    def vari_size_fitting(self,x_b, x_e, y_b, y_e, d):
        x_b = int(x_b)
        x_e = int(x_e)
        y_b = int(y_b)
        y_e = int(y_e)
        filter_size_x = x_e - x_b
        filter_size_y = y_e - y_b
        if filter_size_x <= 0 or filter_size_y <= 0:
            return
        gener_size = min(filter_size_x, filter_size_y) + 2*d
        gener_size = int(gener_size)

        begin = self.calc_begin(x_b, x_e, y_b, y_e, d, gener_size)
        gener_x_b, filter_x_b, gener_y_b, filter_y_b = begin
        per_max = self.calc_permax(int(gener_x_b), int(gener_x_b + gener_size),
                                   int(gener_y_b), int(gener_y_b + gener_size))

        if per_max >= self.per_input or filter_size_x == 1 or filter_size_y == 1:
            x = []
            y = []
            z = []

            filter_x_b = 0
            filter_y_b = 0

            for i in range(0, gener_size):
                for j in range(0, gener_size):
                    x.append(i)
                    y.append(j)
                    z.append(self.image[int(gener_x_b + i), int(gener_y_b + j)])
            p = fitting.fitting(x, y, z)

            for i in range(0, filter_size_x):
                for j in range(0, filter_size_y):
                    xi = int(filter_x_b + i)
                    yi = int(filter_y_b + j)
                    zi = fitting.func(xi, yi, p)
                    x_real = int(x_b + i)
                    y_real = int(y_b + j)
                    self.refer_img[x_real, y_real] = zi

            
        else:
            if d <= 1:
                d = 1
            else:
                d = round(d/2)
            if filter_size_x > 1:
                filter_size_x_1 = round(filter_size_x/2)
            else:
                filter_size_x_1 = 1
            if filter_size_y > 1:
                filter_size_y_1 = round(filter_size_y/2)
            else:
                filter_size_y_1 = 1

            self.vari_size_fitting(x_b, x_b + filter_size_x_1,
                             y_b, y_b + filter_size_y_1, d)
            self.vari_size_fitting(x_b + filter_size_x_1, x_e,
                             y_b, y_b + filter_size_y_1, d)
            self.vari_size_fitting(x_b, x_b + filter_size_x_1,
                             y_b+filter_size_y_1, y_e, d)
            self.vari_size_fitting(x_b + filter_size_x_1, x_e,
                             y_b + filter_size_y_1, y_e, d)

        filter = np.zeros((filter_size_x, filter_size_y))
        filter2 = np.zeros((filter_size_x, filter_size_y))

        for i in range(0, filter_size_x):
            for j in range(0, filter_size_y):
                x_real = int(x_b + i)
                y_real = int(y_b + j)
                zi = self.refer_img[x_real, y_real]
                self.diff2[x_real, y_real] = abs(zi
                                        - self.image[x_real, y_real]) ** 2
                filter[i, j] = abs(zi - self.image[x_real, x_real])
                filter2[i, j] = abs(zi - self.image[x_real, x_real]) ** 2
        var_diff_in_filter = np.var(filter)
        mean_diff2_in_filter = np.mean(filter2)

        for i in range(0, filter_size_x):
            for j in range(0, filter_size_y):
                x_real = int(x_b + i)
                y_real = int(y_b + j)
                self.delta[x_real, y_real] = mean_diff2_in_filter
                self.lamda[x_real, y_real] = var_diff_in_filter

    def gener_refer_img(self):
        h, w = self.image.shape
        q_h = int(h/self.filter_size)
        q_w = int(w/self.filter_size)
        r_h = h % self.filter_size
        r_w = w % self.filter_size

        hs = np.zeros((q_h, 2))
        ws = np.zeros((q_w, 2))

        begin =0
        for i_h in range(0, q_h):
            if i_h >= q_h - r_h:
                hs[i_h][0] = int(self.filter_size+1)
                hs[i_h][1] = begin
            else:
                hs[i_h][0] = int(self.filter_size)
                hs[i_h][1] = begin
            begin = begin + hs[i_h][0]
        begin = 0
        for i_w in range(0, q_w) :
            if i_w >= q_w - r_w :
                ws[i_w][0] = int(self.filter_size+1)
                ws[i_w][1] = begin
            else:
                ws[i_w][0] = int(self.filter_size)
                ws[i_w][1] = begin
            begin = begin + ws[i_w][0]

        d = self.paddle_size

        for i_h in range(0, q_h):
            x_b = hs[i_h][1]
            x_e = hs[i_h][1] + hs[i_h][0]
            for i_w in range(0, q_w):
                y_b = ws[i_w][1]
                y_e = ws[i_w][1] + ws[i_w][0]
                self.vari_size_fitting(x_b, x_e, y_b, y_e, d)

        self.eWeight = self.diff2/(self.lamda*self.delta)
        self.filter_img = (self.image + self.eWeight * self.refer_img) / (1 + self.eWeight)

        im2 = Image.fromarray(self.filter_img)
        plt.imshow(im2, cmap=plt.cm.gray)
        plt.show()
        im1 = Image.fromarray(self.refer_img)
        plt.imshow(im1, cmap=plt.cm.gray)
        plt.show()
        self.refer_x = self.refer_img.flatten().astype('float')
        im = Image.fromarray(self.image)
        plt.imshow(im, cmap=plt.cm.gray)
        plt.show()

    def initial_U(self):
        U = np.zeros((self.numPixels, self.n_clusters))
        idx = np.arange(self.numPixels)
        for ii in range(self.n_clusters):
            idxii = idx % self.n_clusters == ii
            U[idxii, ii] = 1
        return U

    def update_U(self):
        '''Compute weights'''
        c_mesh1, idx_mesh = np.meshgrid(self.C, self.X)
        c_mesh2, refer_idx_mesh = np.meshgrid(self.C, self.refer_x)
        power = 1. / (self.m - 1)
        p1 = (abs(idx_mesh - c_mesh1) ** 2 +
              self.alpha * (abs(refer_idx_mesh - c_mesh2) ** 2)) ** power
        p2 = np.sum((1. / (abs(idx_mesh - c_mesh1) ** 2 +
              self.alpha * (abs(refer_idx_mesh - c_mesh2) ** 2))
                     ** power), axis=1)

        return 1. / (p1 * p2[:, None])

    def update_C(self):
        '''Compute centroid of clusters'''
        numerator = np.dot(self.X + self.alpha * self.refer_x, self.U ** self.m)
        denominator = np.sum((self.U ** self.m) * (1 + self.alpha), axis=0)
        return numerator / denominator

    def form_clusters(self):
        '''Iterative training'''
        d = 100
        self.U = self.initial_U()
        if self.max_iter != -1:
            i = 0
            while True:
                self.C = self.update_C()
                old_u = np.copy(self.U)
                self.U = self.update_U()
                d = np.sum(abs(self.U - old_u))
                print("Iteration %d : cost = %f" % (i, d))

                if d < self.epsilon or i > self.max_iter:
                    break
                i += 1
        else:
            i = 0
            while d > self.epsilon:
                self.C = self.update_C()
                old_u = np.copy(self.U)
                self.U = self.update_U()
                d = np.sum(abs(self.U - old_u))
                print("Iteration %d : cost = %f" % (i, d))

                if d < self.epsilon or i > self.max_iter:
                    break
                i += 1
        self.segmentImage()

    def deFuzzify(self):
        return self.C[np.argmax(self.U, axis=1)]

    def segmentImage(self):
        '''Segment image based on max weights'''

        result = self.deFuzzify()
        self.result = result.reshape(self.shape).astype('int')

        return self.result







if __name__ == '__main__':

    target_img_path = "/Users/wanxiaoguang/Downloads/fuzzy/pythonProject/FCMVMF/vmf-synthetic.png"
    img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
    fcm_cluster = FastFcm(img, image_bit=2, n_clusters=2, m=2, max_iter=100,
                      epsilon=0.00001, reduction=1, kernel_size=5,
                      neighbour_effect=3)›
    fcm_cluster.form_clusters()
    fcm_result = fcm_cluster.result
    cluster = FCMVMF(img, n_clusters=2, m=2, max_iter=20, epsilon=0.00001, fcm_result=fcm_result, per_input=0.8, filter_size=2, gener_size =4, alpha= 200)
    cluster.gener_refer_img()
    cluster.form_clusters()
    result = cluster.result

    plt.imshow(result, cmap=plt.cm.gray)
    plt.show()


