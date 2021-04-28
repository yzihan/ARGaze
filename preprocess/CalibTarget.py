#!/usr/bin/env python
import cv2
import numpy as np

class Validater:
    kernel: np.ndarray
    background: np.ndarray

    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        self.background = None

    def firstpass(self, frame: np.ndarray, frame_all: int):
        acc_frame = frame.astype(np.float) * (1. / frame_all)
        self.background = acc_frame if self.background is None else (self.background + acc_frame)

    def try_load_cache(self, prefix: str):
        self.background = None
        try:
            self.background = np.load(prefix + '.calib.npy')
            return True
        except:
            return False

    def save_cache(self, prefix: str):
        np.save(prefix + '.calib.npy', self.background)
    
    def preprocess(self, frame: np.ndarray):
        diff = np.clip(np.abs(frame - self.background.astype(np.float)), 0, 255).astype(np.uint8)
        edge = cv2.Canny(diff, 100, 100)
        density = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, self.kernel, iterations=5)
        density = cv2.morphologyEx(density, cv2.MORPH_OPEN, self.kernel, iterations=1)

        return [density, diff, edge]
    
    def validate(self, ctx: [np.ndarray], x: int, y: int, r: (np.ndarray, np.ndarray)) -> bool:
        return True


class Analyzer(Validater):

    def __init__(self):
        super(Analyzer, self).__init__()

    def analyze(self, frame: np.ndarray, dump: bool = False):
        ctx = self.preprocess(frame)

        __contours = cv2.findContours(ctx[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(__contours) == 2:
            contours, __hierarchy = __contours
        else:
            _, contours, __hierarchy = __contours

        cnt = 0
        pos = 0, 0
        r = None
        for contour in contours:
            area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)
    #        approx = cv2.approxPolyDP(contour, 40, True)
            _pos, (w, h), _ = rect
            rectArea = w * h
            if rectArea * 0.7 <= area and area >= 40*40 and w * 0.4 >= (w - h):
                cnt = 1
                pos = np.int64(np.around(_pos))
                r = rect, contour

        if dump:
            return cnt, pos, r, ctx
        else:
            return cnt, pos, r
    
    def render(self, frame: np.ndarray, pos: (int, int), r: (np.ndarray, np.ndarray)):
        rect, contour = r
        box = np.int64(np.around(cv2.boxPoints(rect)))
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 4)
        cv2.drawContours(frame, [contour], 0, (255, 0, 0), 4)
        cv2.circle(frame, pos, 2, (255, 0, 255), 10)


import sys
fn = 'cut2.mp4'

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
                    frame = frame[720:1440,0:1080]
                    ctx.firstpass(frame, frame_all // int(fps))
            ctx.save_cache(fn)
            print('{:.2%}  '.format(1.))
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)   # reset position

    while True:
        succeeded, frame = video.read()
        if not succeeded:
            break
        o = cv2.waitKey(1)
        if o == 27:
            break
        if o == 97:
            while cv2.waitKey(0) != 97:
                pass
        frame = frame[720:1440,0:1080]

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
