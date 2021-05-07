# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Finds and highlights lane lines in dashboard camera videos.
See README.md for more info.

Author: Peter Moran
Created: 8/1/2017
"""
import time
import glob
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append('/usr/local/lib')
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import symfit
from imageio.core import NeedDownloadError
from natsort import natsorted

#from dynamic_subplot import DynamicSubplot
from windows_chung import Window, filter_window_list, joint_sliding_window_update, window_image, sliding_window_update

# Import moviepy and install ffmpeg if needed.
try:
    from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips
except NeedDownloadError as download_err:
    if 'ffmpeg' in str(download_err):
        prompt = input('The dependency `ffmpeg` is missing, would you like to download it? [y]/n')
        if prompt == '' or prompt == 'y' or prompt == 'Y':
            from imageio.plugins import ffmpeg

            ffmpeg.download()
            from moviepy.editor import VideoFileClip
        else:
            raise download_err
    else:
        # Unknown download error
        raise download_err

REGULATION_LANE_WIDTH = 3.7


class DashboardCamera:
    def __init__(self, chessboard_img_fnames, chessboard_size, lane_shape, scale_correction=(30 / 720, 3.7 / 700)):
        # Get image size
        example_img = cv2.imread(chessboard_img_fnames[0])
        self.img_size = example_img.shape[0:2]
        self.img_height = self.img_size[0]
        self.img_width = self.img_size[1]

        # Calibrate
        self.camera_matrix, self.distortion_coeffs = self.calibrate(chessboard_img_fnames, chessboard_size)
        # Define overhead transform and its inverse
        top_left, top_right, bottom_left, bottom_right = lane_shape
        source = np.float32([top_left, top_right, bottom_right, bottom_left])
        destination = np.float32([(bottom_left[0], 0), (bottom_right[0], 0),
                                  (bottom_right[0], self.img_height - 1), (bottom_left[0], self.img_height - 1)])
        self.overhead_transform = cv2.getPerspectiveTransform(source, destination)
        self.inverse_overhead_transform = cv2.getPerspectiveTransform(destination, source)
        self.y_m_per_pix = scale_correction[0]
        self.x_m_per_pix = scale_correction[1]

    def calibrate(self, chessboard_img_files, chessboard_size):
 
        # Create placeholder lists
        chess_rows, chess_cols = chessboard_size
        chess_corners = np.zeros((chess_cols * chess_rows, 3), np.float32)
        chess_corners[:, :2] = np.mgrid[0:chess_rows, 0:chess_cols].T.reshape(-1, 2)
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Determine object point, image point pairs
        for fname in chessboard_img_files:
            # Load images
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            found, img_corners = cv2.findChessboardCorners(gray, (chess_rows, chess_cols), None)

            # If found, save object points, image points
            if found:
                objpoints.append(chess_corners)
                imgpoints.append(img_corners)

        # Perform calibration
        success, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.img_size,
                                                                                None, None)
        if not success:
            raise Exception("Camera calibration unsuccessful.")
        return camera_matrix, dist_coeffs

    def undistort(self, image):
        """
        Removes distortion this camera's raw images.
        """
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs, None, self.camera_matrix)

    def warp_to_overhead(self, undistorted_img):
        """
        Transforms this camera's images from the dashboard perspective to an overhead perspective.

        Note: Make sure to undistort first.
        """
        return cv2.warpPerspective(undistorted_img, self.overhead_transform, dsize=(self.img_width, self.img_height))

    def warp_to_dashboard(self, overhead_img):
        """
        Transforms this camera's images from an overhead perspective back to the dashboard perspective.
        """
        return cv2.warpPerspective(overhead_img, self.inverse_overhead_transform,
                                   dsize=(self.img_width, self.img_height))


class LaneFinder:
    def __init__(self, cam: DashboardCamera, window_shape=(10, 10), search_margin=50, max_frozen_dur=15):
        
        self.camera = cam # 
 
        # Create windows
        self.Lines = []
        self.valid_Lines = []
        self.valid_windows_num = []
        self.line_num = 3
        self.window_num = 18
        # Line States
        self.Is_Dashed = [False,False,False]
        self.Is_valid = [True, True, True]
        # Set Lines
        self.line_left = None
        self.line_mid = None
        self.line_right = None
        self.left_arg = None
        self.mid_arg = None
        self.right_arg = None
        # line variable
        self.mid_line = 160
        # Genarate window sets
        window_set = []
        for i in range(self.line_num):
            for j in range(self.window_num): # window_num
                x_init = cam.img_width /2 #
                window_set.append(Window(window_shape, cam.img_size, x_init, max_frozen_dur))
            self.Lines.append(list(window_set)) #call by value
            window_set.clear()
            
        self.search_margin = search_margin

        # Initialize visuals
        VIZ_OPTIONS = ('dash_undistorted', 'overhead', 'lab_b', 'lab_b_binary', 'lightness', 'lightness_binary',
                       'value', 'value_binary', 'pixel_scores', 'windows_raw', 'windows_filtered', 'highlighted_lane',
                       'presentation')
        self.visuals = {name: None for name in VIZ_OPTIONS}  # Storage location of visualization images
        self.__viz_desired = None  # The visuals we want to save
        self.__viz_dependencies = {'windows_raw': ['pixel_scores'],  # Dependencies of visuals on other visuals
                                   'windows_filtered': ['pixel_scores'],
                                   'presentation': ['highlighted_lane', 'overhead', 'windows_raw', 'windows_filtered',
                                                    'pixel_scores']}

    def find_lines(self,img_dashboard, last_mid_pos, visuals=None, Mode = 'test'):
        
        # Account for visualization options
        if visuals is None:
            visuals = ['highlighted_lane']
        self.__viz_desired = self.viz_fix_dependencies(visuals)

        # Undistort and transform to overhead view
        img_dash_undistorted = self.camera.undistort(img_dashboard)
        img_overhead = self.camera.warp_to_overhead(img_dash_undistorted)
        '''
        plt.figure(2)
        plt.imshow(img_overhead)
        plt.show()'''

        # Score pixels
        pixel_scores = self.score_pixels(img_overhead)

        #print("init whindows, a:{0},b:{1},c:{2}".format(len(self.Lines[0]), len(self.Lines[1]) ,len(self.Lines[2])))

        # Score Image copy
        score_img = np.array(pixel_scores)

        # Line detect
        for i in range(self.line_num):
            self.Is_Dashed[i], self.Is_valid[i] = sliding_window_update(self.Lines[i], score_img, self.window_num, min_activity = 0.2)
            print("Line {0} is Valid: {1}, Dashed:{2}".format(i, self.Is_valid[i], self.Is_Dashed[i]))
        # Check Valid windows
        for i in range(self.line_num):
            valid_windows, valid_len = filter_window_list(self.Lines[i], remove_frozen=False, remove_dropped=True)
            self.valid_Lines.append(list(valid_windows))
            self.valid_windows_num.append(int(valid_len))
        # Check num of the windows
        for i in range(self.line_num):
            if self.valid_windows_num[i] < 3:
                self.Is_valid[i] = False
        #print("valid windows, a:{0},b:{1},c:{2}".format(self.valid_windows_num[0], self.valid_windows_num[1], self.valid_windows_num[2]))
        #print("valid lanes: {0}, {1}, {2}".format(self.Is_valid[0],self.Is_valid[1],self.Is_valid[2]))


        fit_vals = []
        for i in range(self.line_num):
            if self.Is_valid[i]:
                fit_vals.append(self.fit_lanes_v2(zip(*[window.pos_xy() for window in self.valid_Lines[i]])))
            else:
                fit_vals.append(None)

        # Determine the location of the polynomial fit line for each row of the image
        y_fit = []
        for i in range(self.line_num):
            if self.Is_valid[i]:
                y_fit.append(np.array(range(self.valid_Lines[i][self.valid_windows_num[i] - 1].y_begin, self.valid_Lines[i][0].y_end)))
            else:
                y_fit.append(None)

        x_fit = []
        for i in range(self.line_num):
            if self.Is_valid[i]:
                x_fit.append(fit_vals[i]['a2'] * y_fit[i] ** 2 + fit_vals[i]['a1'] * y_fit[i] + fit_vals[i]['a0'])
            else:
                x_fit.append(None)

        # Calculate Curvatures, (output-> negative:left, positive: right)
        curv = []
        for i in range(self.line_num):
            if self.Is_valid[i]:
                curv.append(self.calc_curvature_v2(self.valid_windows_num[i] - 1, self.valid_Lines[i]))
        for m in curv:
            print(m)
        curv_avg = 0
        for i in range(len(curv)):
            curv_avg += curv[i]/len(curv)
            #print(" curv[{0}]: {1}".format(i,curv[i]))
        print(" [curv_avg: {0}]".format(curv_avg))

        # Instants
        r_mid_pos = 160 #reference mid lane position
        lane_half_width = 180 #half width of lane
        lane_detect_height = 120
        # Variables
        self.mid_line = 160  # mid line position

        # Determind line position
        for i in range(self.line_num):
            if self.Is_valid[i]:
                if self.Is_Dashed[i]:
                    self.line_mid = self.Lines[i]
                    self.mid_arg = i
                elif self.line_mid is None: # is not dashed, and if no mid_line is detected
                    if self.Lines[i][0].pos_xy()[0] + (self.Lines[i][0].pos_xy()[1]-lane_detect_height) * curv_avg <= last_mid_pos :
                        self.line_left = self.Lines[i]
                        self.left_arg = i
                    elif self.Lines[i][0].pos_xy()[0] + (self.Lines[i][0].pos_xy()[1]-lane_detect_height) * curv_avg >= last_mid_pos :
                        self.line_right = self.Lines[i]
                        self.right_arg = i
                elif self.line_mid is not None:
                    if Mode == 'Run':
                        break
                    elif self.Lines[i][0].pos_xy()[0] + (self.Lines[i][0].pos_xy()[1]-lane_detect_height) * curv_avg <= self.line_mid[0].pos_xy()[0] :
                        self.line_left = self.Lines[i]
                        self.left_arg = i
                    elif self.Lines[i][0].pos_xy()[0] + (self.Lines[i][0].pos_xy()[1]-lane_detect_height) * curv_avg  >= self.line_mid[0].pos_xy()[0]:
                        self.line_right = self.Lines[i]
                        self.right_arg = i

                    
        print("left: {0}, mid: {1}, right: {2}".format(self.left_arg, self.mid_arg, self.right_arg))
        # Determind mid_line Value, neg:left, pos:right

        if self.line_mid is not None: 
            #print("<mid-line based mid_line calculate>")
            self.mid_line = self.line_mid[0].pos_xy()[0] + (self.line_mid[0].pos_xy()[1]-lane_detect_height) * curv_avg
            #print(" mid_line:{0}".format(mid_line))
            print("check mid line")

        elif self.line_left is not None:
            self.mid_line = self.line_left[0].pos_xy()[0] + lane_half_width + (self.line_left[0].pos_xy()[1]-lane_detect_height) * curv_avg
            print("check left line")

        elif self.line_right is not None:
            self.mid_line = self.line_right[0].pos_xy()[0] - lane_half_width  + (self.line_right[0].pos_xy()[1]-lane_detect_height) * curv_avg
            print("check right line")
        else:
            self.mid_line = last_mid_pos
            print("no line")
        lane_position = self.mid_line
        
        # dodge if over 2 line
        no_dash_num = 0
        for i in range(self.line_num):
            if self.Is_valid[i]:
                if not self.Is_Dashed[i]:
                    no_dash_num += 1
        if no_dash_num >= 3:
            self.mid_line = last_mid_pos

        print("mid_line: {0}".format(self.mid_line))

        if Mode == 'Run':
            return self.mid_line


        # Log visuals
        self.viz_save('dash_undistorted', img_dash_undistorted)
        self.viz_save('overhead', img_overhead)
        self.viz_save('pixel_scores', pixel_scores)
        self.viz_save('windows_raw', self.visuals['pixel_scores'],
                      img_proc_func = lambda img : self.viz_windows(img, 'raw'))
        self.viz_save('windows_filtered', self.visuals['pixel_scores'],
                      img_proc_func = lambda img : self.viz_windows(img, 'filtered')) 
        self.viz_save('highlighted_lane', img_dash_undistorted,
                      img_proc_func = lambda img : self.viz_lane(img, self.camera, x_fit, y_fit, active_lane='left'))
        self.viz_save('presentation', self.visuals['highlighted_lane'],
                      img_proc_func = lambda img : self.viz_presentation(img, lane_position, curv_avg))

        return y_fit, x_fit, self.Is_valid


    def score_pixels(self, img) -> np.ndarray:
        
        # Settings to run thresholding operations on
        settings = [{'name': 'lab_b', 'cspace': 'LAB', 'channel': 2, 'clipLimit': 2.0, 'threshold': 50}, #150
                    {'name': 'value', 'cspace': 'HSV', 'channel': 2, 'clipLimit': 6.0, 'threshold': 55}, #220
                    {'name': 'lightness', 'cspace': 'HLS', 'channel': 1, 'clipLimit': 2.0, 'threshold': 60}] #210

        # Perform binary thresholding according to each setting and combine them into one image.
        scores = np.zeros(img.shape[0:2]).astype('uint8')
        '''
        fig = plt.figure(3)
        ax = []
        tmp = 1'''
        for params in settings[1:]:
            # Change color space
            color_t = getattr(cv2, 'COLOR_RGB2{}'.format(params['cspace']))

            imgUMat = cv2.UMat(img)

            gray = cv2.cvtColor(imgUMat, color_t).get()
            gray2 = gray[:, :, params['channel']]

            # Normalize regions of the image using CLAHE
            clahe = cv2.createCLAHE(params['clipLimit'], tileGridSize=(8, 8))
            norm_img = clahe.apply(gray2)
            #norm_img = gray

            # Threshold to binary
            ret, binary = cv2.threshold(norm_img, params['threshold'], 1, cv2.THRESH_BINARY_INV)

            scores += binary
            '''
            ax.append(fig.add_subplot(3,2,tmp))
            ax[tmp-1].imshow(norm_img)
            tmp += 1
            ax.append(fig.add_subplot(3,2,tmp))
            ax[tmp-1].imshow(binary)
            tmp += 1'''
            # Save images
            self.viz_save(params['name'], gray2)
            self.viz_save(params['name'] + '_binary', binary)
        #plt.show()
        '''
        plt.figure(4)
        plt.imshow(cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX))
        plt.show()'''
        return cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX)

    def fit_lanes(self, points_left, points_right, fit_globally=False) -> dict:
        
        xl, yl = points_left
        xr, yr = points_right

        fit_vals = {}
        if fit_globally:
            # Define global model to fit
            x_left, y_left, x_right, y_right = symfit.variables('x_left, y_left, x_right, y_right')
            a, b, x0_left, x0_right = symfit.parameters('a, b, x0_left, x0_right')

            model = symfit.Model({
                x_left: a * y_left ** 2 + b * y_left + x0_left,
                x_right: a * y_right ** 2 + b * y_right + x0_right
            })

            # Apply fit
            xl, yl = points_left
            xr, yr = points_right
            fit = symfit.Fit(model, x_left=xl, y_left=yl, x_right=xr, y_right=yr)
            fit = fit.execute()
            fit_vals.update({'ar': fit.value(a), 'al': fit.value(a), 'bl': fit.value(b), 'br': fit.value(b),
                             'x0l': fit.value(x0_left), 'x0r': fit.value(x0_right)})

        else:
            # Fit lines independently
            x, y = symfit.variables('x, y')
            a, b, x0 = symfit.parameters('a, b, x0')

            model = symfit.Model({
                x: a * y ** 2 + b * y + x0,
            })

            # Apply fit on left
            fit = symfit.Fit(model, x=xl, y=yl)
            fit = fit.execute()
            fit_vals.update({'al': fit.value(a), 'bl': fit.value(b), 'x0l': fit.value(x0)})

            # Apply fit on right
            fit = symfit.Fit(model, x=xr, y=yr)
            fit = fit.execute()
            fit_vals.update({'ar': fit.value(a), 'br': fit.value(b), 'x0r': fit.value(x0)})

        return fit_vals

    def fit_lanes_v2(self, points) -> dict:
        xl, yl = points

        fit_vals = {}
        # Fit lines independently
        x, y = symfit.variables('x, y')
        a, b, x0 = symfit.parameters('a, b, x0')

        model = symfit.Model({
            x: a * y ** 2 + b * y + x0,
        })

        # Apply fit on left
        fit = symfit.Fit(model, x=xl, y=yl)
        fit = fit.execute()
        fit_vals.update({'a2': fit.value(a), 'a1': fit.value(b), 'a0': fit.value(x0)})

        return fit_vals

    def calc_curvature(self, windows: List[Window]):
        
        x, y = zip(*[window.pos_xy() for window in windows])
        x = np.array(x)
        y = np.array(y)
        fit_cr = np.polyfit(y * camera.y_m_per_pix, x * camera.x_m_per_pix, 2)
        #fit_cr = np.polyfit(y, x, 2)
        print("c2: {0}, c1: {1}, c0: {2}".format(fit_cr[0],fit_cr[1],fit_cr[2]))
        y_eval = np.max(y)
        return ((1 + (2 * fit_cr[0] * y_eval * camera.y_m_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])

    def calc_curvature_v2(self, max_arg, windows: List[Window]):
        [top_x,top_y] = windows[max_arg].pos_xy()
        [bottom_x, bottom_y] = windows[0].pos_xy()
        return -(bottom_x - top_x)/(bottom_y-top_y)

    def viz_save(self, name, image, img_proc_func=None):
       
        if 'all' not in self.__viz_desired and name not in self.__viz_desired:
            return  # Don't save this image
        if img_proc_func is not None:
            image = img_proc_func(image)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise Exception('Image is not 3 channels or could not be converted to 3 channels. Cannot use.')
        self.visuals[name] = image

    def viz_fix_dependencies(self, viz_names: list):
        
        for viz_opt in self.__viz_dependencies:
            if viz_opt in viz_names:
                for dependency in self.__viz_dependencies[viz_opt]:
                    viz_names.append(dependency)
        return viz_names

    def viz_presentation(self, lane_img, lane_position, curve_radius, lane_width=REGULATION_LANE_WIDTH):
       
        presentation_img = np.copy(lane_img)
        lane_position_prcnt = lane_position / lane_width

        # Show overlays
        overhead_img = cv2.resize(self.visuals['overhead'], None, fx=1 / 3.0, fy=1 / 3.0)
        titled_overlay(presentation_img, overhead_img, 'Overhead (not to scale)', (0, 0))
        overhead_img = cv2.resize(self.visuals['windows_raw'], None, fx=1 / 3.0, fy=1 / 3.0)
        titled_overlay(presentation_img, overhead_img, 'Raw Lane Detection', (presentation_img.shape[1] // 3, 0))
        overhead_img = cv2.resize(self.visuals['windows_filtered'], None, fx=1 / 3.0, fy=1 / 3.0)
        titled_overlay(presentation_img, overhead_img, 'Filtered Lane Detection', (presentation_img.shape[1] // 3 * 2, 0))

        # Show position
        x_text_start, y_text_start = (10, 350)
        line_start = (10 + x_text_start, 40 + y_text_start)
        line_len = 300
        cv2.putText(presentation_img, "Position", org=(x_text_start, y_text_start), fontScale=2, thickness=3,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, lineType=cv2.LINE_AA, color=(255, 255, 255))
        cv2.line(presentation_img, color=(255, 255, 255), thickness=2,
                 pt1=(line_start[0], line_start[1]),
                 pt2=(line_start[0] + line_len, line_start[1]))
        cv2.circle(presentation_img, center=(line_start[0] + int(lane_position_prcnt * line_len), line_start[1]),
                   radius=8,
                   color=(255, 255, 255))
        cv2.putText(presentation_img, '{:.2f} m'.format(lane_position), fontScale=1, thickness=1,
                    org=(line_start[0] + int(lane_position_prcnt * line_len) + 5, line_start[1] + 35),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), lineType=cv2.LINE_AA)

        # Show radius of curvature
        cv2.putText(presentation_img, "Curvature = {:>4.0f} m".format(curve_radius), fontScale=1, thickness=2,
                    org=(x_text_start, 130 + y_text_start), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255),
                    lineType=cv2.LINE_AA)

        return presentation_img

    def viz_windows(self, score_img, mode):
        """Displays the position of the windows over a score image."""
        win_imgs = []
        if mode == 'filtered':
            for i in range(self.line_num):
                if self.Is_valid[i]:
                    win_imgs.append(window_image(self.Lines[i], 'x_filtered', color=(0, 255, 0)))
        elif mode == 'raw':
            color = (255, 0, 0)
            for i in range(self.line_num):
                if self.Is_valid[i]:
                    win_detected, arg = filter_window_list(self.Lines[i], False, False, remove_undetected=True)
                    win_imgs.append(window_image(win_detected, 'x_measured', color, color, color))
        else:
            raise Exception('mode is not valid')
        combined = np.zeros((self.camera.img_height, self.camera.img_width, 3))
        for i in range(len(win_imgs)):
            combined = combined + win_imgs[i]
        combined_ = np.array(combined).astype(np.uint8)
        return cv2.addWeighted(score_img, 1, combined_, 0.5, 0)

    def viz_lane(self, undist_img, camera, fit_x, fit_y, active_lane = 'right'):
       
        # Create an undist_img to draw the lines on
        lane_poly_overhead = np.zeros_like(undist_img).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        #pts_left = np.array(np.transpose(np.vstack([left_fit_x, fit_y_a])))
        #pts_right = np.array(np.flipud(np.transpose(np.vstack([right_fit_x, fit_y_b]))))
        #pts = np.vstack((pts_left, pts_right)
        pts = np.empty(0).reshape(0,2)
        if self.line_mid is not None:
            pts_mid = np.array(np.transpose(np.vstack([ fit_x[self.mid_arg], fit_y[self.mid_arg] ])))
            pts = np.vstack((pts_mid))
        if active_lane == 'right':
            if self.line_right is not None:
                pts_right = np.array(np.flipud(np.transpose(np.vstack([ fit_x[self.right_arg], fit_y[self.right_arg] ]))))
                pts = np.vstack((pts, pts_right))
            else:
                pts = np.vstack((pts, [self.camera.img_width, self.camera.img_height]))
        if active_lane == 'left':
            if self.line_left is not None:
                pts_left = np.array(np.flipud(np.transpose(np.vstack([ fit_x[self.left_arg], fit_y[self.left_arg] ]))))
                pts = np.vstack((pts, pts_left))
            else:
                pts = np.vstack((pts, [0, self.camera.img_height]))


        # Draw the lane onto the warped blank undist_img
        cv2.fillPoly(lane_poly_overhead, np.int_([pts]), (0, 255, 0))

        # Warp back to original undist_img space
        lane_poly_dash = camera.warp_to_dashboard(lane_poly_overhead)

        # Combine the result with the original undist_img
        return cv2.addWeighted(undist_img, 1, lane_poly_dash, 0.3, 0)

    def viz_pipeline(self, img, last_mid_pos): #for image
        """Displays most of the steps in the image processing pipeline for a single image."""
        # Find line
        y_fit, x_fit, valid = self.find_lines(img,last_mid_pos, ['all'], 'test') #이부분 import
        #print("find_lines complete")
        # Plot

        plt.subplot(2,2,1), plt.imshow(self.visuals['dash_undistorted'])
        plt.subplot(2,2,2), plt.imshow(self.visuals['overhead'], cmap='gray')
        plt.subplot(2,2,3), plt.imshow(self.visuals['pixel_scores'], cmap='gray')
        plt.subplot(2,2,4), plt.imshow(self.visuals['highlighted_lane'])
        plt.show()
        '''
        dynamic_subplot = DynamicSubplot(3, 4)
        dynamic_subplot.imshow(self.visuals['dash_undistorted'], "Undistorted Road")
        dynamic_subplot.imshow(self.visuals['overhead'], "Overhead", cmap='gray')
        dynamic_subplot.imshow(self.visuals['lightness'], "Lightness", cmap='gray')
        dynamic_subplot.imshow(self.visuals['lightness_binary'], "Binary Lightness", cmap='gray')
        dynamic_subplot.skip_plot()
        dynamic_subplot.skip_plot()
        dynamic_subplot.imshow(self.visuals['value'], "Value", cmap='gray')
        dynamic_subplot.imshow(self.visuals['value_binary'], "Binary Value", cmap='gray')
        dynamic_subplot.imshow(self.visuals['pixel_scores'], "Scores", cmap='gray')
        dynamic_subplot.imshow(self.visuals['windows_raw'], "Selected Windows")
        dynamic_subplot.imshow(self.visuals['windows_raw'], "Fitted Lines", cmap='gray')
        for i in range(3):
            if valid[i]:
                dynamic_subplot.modify_plot('plot', x_fit[i], y_fit[i])
        dynamic_subplot.modify_plot('set_xlim', 0, camera.img_width)
        dynamic_subplot.modify_plot('set_ylim', camera.img_height, 0)
        dynamic_subplot.imshow(self.visuals['highlighted_lane'], "Highlighted Lane")'''

    def get_mid_line(self, img, last_mid_pos):
        return self.find_lines(img,last_mid_pos, None, 'Run')


def titled_overlay(image, overlay, title, org, border_thickness=2):
    """Puts a title above the overlay image and places it in image at the given origin."""
    # Place title
    title_img = np.ones((50, overlay.shape[1], 3)).astype('uint8') * 255
    cv2.putText(title_img, title, org=(10, 35), fontScale=1, thickness=2, color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, lineType=cv2.LINE_AA)

    # Add title to overlay
    overlay = np.concatenate((title_img, overlay), axis=0)

    # Add border to overlay
    overlay[:border_thickness, :, :] = 255
    overlay[-border_thickness:, :, :] = 255
    overlay[:, :border_thickness, :] = 255
    overlay[:, -border_thickness:, :] = 255

    # Place overlay onto image
    x_offset, y_offset = org
    image[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1]] = overlay

    # Add a white borderRemoving path from Python search module path - Stack Overflow


if __name__ == '__main__':
    argc = len(sys.argv)
    # Calibrate using checkerboard
    calibration_img_files = glob.glob('./data/camera_cal_low/*.jpg')
    #lane_shape = [(584, 458), (701, 458), (295, 665), (1022, 665)]
    #lane_shape = [(150, 320), (1060, 320), (0, 700), (1212, 700)]
    lane_shape = [(55, 80), (265, 80), (0, 165), (320, 165)] #곡선
    camera = DashboardCamera(calibration_img_files, chessboard_size=(9, 6), lane_shape=lane_shape)

    # laod last mid pos
    last_mid_pos = 0
    file_mid_log = open('./data/mid_line_log/mid_log.txt','r')
    while True:
        line = file_mid_log.readline()
        if not line: break
        last_mid_pos = int(line)
    print("last mid-line position: {0}".format(last_mid_pos))
    file_mid_log.close()

    if str(sys.argv[1]) == 'test':
        # Image load
        test_imgs = glob.glob('./data/test_image_set_0/*.jpeg')
        #test_imgs = glob.glob('./data/test_images/*.jpg')
	    # Lane Find on image
        for num, img_file in enumerate(test_imgs[:]):
            print("image num : {0}".format(num))
            img = plt.imread(img_file)
            test_img = cv2.resize(img, dsize=(320, 180), interpolation=cv2.INTER_AREA)
            test_img_copy = np.array(test_img).astype(np.uint8)
            '''
            plt.figure(1)
            plt.imshow(test_img_copy)
            plt.show()'''
            
            lane_finder = LaneFinder(camera)  # need new instance per image to prevent smoothing
            lane_finder.viz_pipeline(test_img_copy, last_mid_pos)
            #print("final mid_line: {0}".format(lane_finder.get_mid_line(test_img_copy, last_mid_pos)))

        # Show all plots
        plt.show()

    elif str(sys.argv[1]) == 'Run':
        # Image load
        test_imgs = glob.glob('./data/test_image_set_0/*.jpeg')
        for img_file in test_imgs[:]:
            start_vect=time.time()
            img = plt.imread(img_file)
            test_img = cv2.resize(img, dsize=(320, 180), interpolation=cv2.INTER_AREA)
            test_img_copy = np.array(test_img).astype(np.uint8)

            lane_finder = LaneFinder(camera)  # need new instance per image to prevent smoothing
            print("mid_line: {0}".format(lane_finder.get_mid_line(test_img_copy, last_mid_pos)))
            print("spent time: {0}".format(time.time() - start_vect))

    elif str(sys.argv[1]) == 'Video':
        img_list = glob.glob('./data/test_image_set_0/*.jpeg')
        img_list_sorted = natsorted(img_list,reverse=False)  # Sort the images

        #clips = [ImageClip(m).set_duration(0.1) for m in img_list_sorted]
        visual = 'highlighted_lane'
        clips = []
        cnt = 0
        for img_file in img_list_sorted:
            cnt += 1
            img = plt.imread(img_file)
            '''plt.figure(1)
            plt.imshow(img)
            plt.show()'''
            lane_finder = LaneFinder(camera)
            highlighted_img, last_mid_pos = lane_finder.video_find_lines(img, last_mid_pos, visual)
            clips.append(ImageClip(highlighted_img).set_duration(0.1))
            print('clip : {}, mid: {}'.format(cnt,last_mid_pos))
        concat_clip = concatenate_videoclips(clips, method="compose")
        concat_clip.write_videofile("test.mp4", fps=10)


        '''
        # Video options
        input_vid_file = str(sys.argv[1])
        visual = str(sys.argv[2]) if argc >= 3 else 'presentation'
        if argc >= 4:
            output_vid_file = str(sys.argv[3])
        else:
            name, ext = input_vid_file.split('/')[-1].split('.')
            name += '_' + visual
            ext = 'mp4'
            output_vid_file = './output/' + name + '.' + ext

        # Create video
        lane_finder = LaneFinder(camera)
        input_video = VideoFileClip(input_vid_file)
        print(input_video)
        output_video = input_video.fl_image(lane_finder.viz_callback(last_mid_pos, visual))
        output_video.write_videofile(output_vid_file, audio=False)'''
