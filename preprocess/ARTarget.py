#!/usr/bin/env python
import cv2
import numpy as np

class Validater:
    kernel: np.ndarray
    background: np.ndarray

    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        self.background = None

    def firstpass(self, frame: np.ndarray, frame_all: int):
        """
        retrive "static" background by averaging all frames
        """
        acc_frame = frame[:,:,1].astype(np.float) * (1. / frame_all)
        self.background = acc_frame if self.background is None else (self.background + acc_frame)

    def try_load_cache(self, prefix: str):
        """
        load "static" background from cache
        """
        self.background = None
        try:
            self.background = np.load(prefix + '.ar.npy')
            return True
        except:
            return False

    def save_cache(self, prefix: str):
        """
        save "static" background to cache
        """
        np.save(prefix + '.ar.npy', self.background)

    def preprocess(self, frame: np.ndarray):
        """
        preprocess the frame and get the processed image
        """
        gray = frame[:,:,1].copy() # green channel as gray

        orig_diff = np.clip(np.abs(self.background - gray.astype(np.float)), 0, 255).astype(np.uint8)
        diff = cv2.morphologyEx(orig_diff, cv2.MORPH_CLOSE, self.kernel, iterations=10)
        _, diff = cv2.threshold(diff, 100, 255, cv2.THRESH_TOZERO)

        return [diff, gray, orig_diff]

    def validate(self, ctx: np.ndarray, x: int, y: int, r: int) -> bool:
        """
        check whether the circle is what we want
        """
        gray = ctx[1]

        if np.average(gray[y - r // 3: y + r // 3, x - r // 3: x + r // 3]) >= 240:
            return True
        return False


class Analyzer(Validater):

    def __init__(self):
        super(Analyzer, self).__init__()

    def analyze(self, frame: np.ndarray, dump: bool = False):
        """
        analyze the frame and get the result
        """
        ctx = self.preprocess(frame)

#        circles = cv2.HoughCircles(ctx[1], cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=5, maxRadius=30)
        circles = cv2.HoughCircles(ctx[0], cv2.HOUGH_GRADIENT, 5, 100, param1=10, param2=10, minRadius=10, maxRadius=18)
#        circles = cv2.HoughCircles(ctx[1], cv2.HOUGH_GRADIENT, 10, 100, param1=300, param2=120, minRadius=10, maxRadius=30)

        cnt = 0
        x, y, r = 0, 0, 0
        if circles is not None:
            circles = np.uint16(np.around(circles[0, :, :]))
            # iterate over all detected circles
            for i in circles[:]:
                _x, _y = i[0], i[1]
                _r = i[2]
                # validate them
                if self.validate(ctx, _x, _y, _r):
                    cnt = cnt + 1
                    x, y, r = _x, _y, _r
        if dump:
            return cnt, (x, y), r, ctx
        else:
            return cnt, (x, y), r

    def render(self, frame: np.ndarray, pos: (int, int), r: int):
        """
        render the detected object
        """
        cv2.circle(frame, pos, r, (0,0,255), 5)
        cv2.circle(frame, pos, 2, (255,0,0), 10)



'''
test program
only be executed when the file is invoked directly, e.g. run: `python ARTarget.py somefile.mp4`
'''

import sys
fn = 'cut1.mp4'

def main():
    global fn
    if len(sys.argv) == 2:
        fn = sys.argv[1]

    ctx = Analyzer()
    video = cv2.VideoCapture(fn)
    if not video.isOpened():
        print('Video device not opened')
        exit(1)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_all = video.get(cv2.CAP_PROP_FRAME_COUNT)

    print('File: {}'.format(fn))
    print('FPS: {}'.format(fps))
    print('Frame Count: {}'.format(frame_all))
    print('Length: {}s'.format(frame_all / fps))

    if 'firstpass' in dir(ctx):
        if not ctx.try_load_cache(fn):
            i = fps
            while True:
                succeeded, frame = video.read()
                if not succeeded:
                    break

                if i > 0:
                    i -= 1
                else:
                    i = fps
                    print('{:.2%}  '.format(video.get(cv2.CAP_PROP_POS_FRAMES) / frame_all), end='\r')
                    frame = frame[720:1440,0:1280]
                    ctx.firstpass(frame, frame_all // int(fps))
            ctx.save_cache(fn)
            print('{:.2%}  '.format(1.))
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)   # reset position

    while True:
        succeeded, frame = video.read()
        print(frame.shape)
        if not succeeded:
            break
        if cv2.waitKey(1) == 27:
            break
        frame = frame[720:1440,0:1280]

        cnt, (x, y), r, dumps = ctx.analyze(frame, dump=True)
        if cnt == 1:
            ctx.render(frame, (x, y), r)

        cv2.imshow('Preview', frame)

        for i, dump in enumerate(dumps):
            if dump is not None:
                cv2.imshow('Dump {}'.format(i), dump)

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
