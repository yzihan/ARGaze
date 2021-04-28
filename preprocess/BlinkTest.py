#!/usr/bin/env python
import os
import cv2
import numpy as np
from typing import *

def clamp(x, a, b):
    return min(max(x,a),b)

CUT_SIZE=512
HALF_CUT_SIZE=CUT_SIZE // 2
USE_LEFT = True
EYE_POS = [363, 679]

class Analyzer:
    cnt: int
    threshold: float
    threval: float
    kernel: np.ndarray
    template: np.ndarray

    prev_blink: int

    eye_cascade: cv2.CascadeClassifier

    eye_pos: Tuple[float,float]
    initial: int

    fixed_eyepos: bool

    def __init__(self, fixed_eyepos: Tuple[int, int] = None):
        self.cnt = 0
        self.threshold = 0.0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        self.accumu = None
        self.template = None

        self.eye_cascade = cv2.CascadeClassifier(os.path.join((cv2.data.haarcascades if 'data' in dir(cv2) else os.path.join(os.path.dirname(__file__), 'haarcascade_data')), 'haarcascade_eye.xml'))

        if fixed_eyepos is None:
            self.fixed_eyepos = False
            self.eye_pos = None
        else:
            self.fixed_eyepos = True
            self.eye_pos = fixed_eyepos
        self.initial = 0

        self.prev_blink = 0

    def firstpass(self, frame: np.ndarray, frame_all: int):
        if self.fixed_eyepos:
            x, y = self.eye_pos
            frame = frame[y - HALF_CUT_SIZE:y + HALF_CUT_SIZE, x - HALF_CUT_SIZE:x + HALF_CUT_SIZE]
        (b, g, r) = cv2.split(frame)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        acc_frame = cv2.merge((b, g, r)).astype(np.float) * (1. / frame_all)
        self.template = acc_frame if self.template is None else (self.template + acc_frame)

    def try_load_cache(self, prefix: str):
        self.template = None
        try:
            self.template = np.load(prefix + '.blink.npy')
            return True
        except:
            return False

    def save_cache(self, prefix: str):
        if self.template is None:
            print('Error! template is none')
            exit(1)
        np.save(prefix + '.blink.npy', self.template)

    def analyze(self, frame: np.ndarray, dump: bool = False):
        blink = False

        if self.fixed_eyepos:
            x, y = self.eye_pos
            frame = frame[y - HALF_CUT_SIZE:y + HALF_CUT_SIZE, x - HALF_CUT_SIZE:x + HALF_CUT_SIZE]
            cut = frame

        (b, g, r) = cv2.split(frame)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        frame0 = cv2.merge((b, g, r))
        frame_cut = frame0.copy()

        frame = np.clip(np.abs(frame0.astype(np.float) - self.template), 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f = frame.astype(np.float32)
        
        if not self.fixed_eyepos:
            gf = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gf)

            if self.eye_pos is not None:
                px, py = self.eye_pos

                x,y = 0,0
                w = 0
                for (ex,ey,ew,eh) in eyes:
                    if ew < CUT_SIZE * 0.4:
                        continue
                    if dump:
                        cv2.rectangle(frame0,(ex,ey),(ex+ew,ey+eh),(255,0,0),5)
                    dist = ((ex - px) ** 2 + (ey - py) ** 2) ** 0.5
                    # if dist < 1000:
                    cw = ew + dist
                    if cw > w:
                        x,y = ex + ew // 2, ey + ew // 2
                        w = cw
            else:
                px, py = 0, 0
                x,y = 0,0
                w = 0
                for (ex,ey,ew,eh) in eyes:
                    if ew < HALF_CUT_SIZE:
                        continue
                    if dump:
                        cv2.rectangle(frame0,(ex,ey),(ex+ew,ey+eh),(255,0,0),5)
                    if ew > w:
                        x,y = ex + ew // 2, ey + ew // 2

            if (x,y) == (0,0):
                if self.eye_pos is not None:
                    x, y = self.eye_pos
            if (x,y) != (0,0):
                if self.initial < 1000:
                    self.initial += 1
                    alpha = 1.0 - 1.0 / self.initial
                    self.eye_pos = ((px - x) * alpha + x, (py - y) * alpha + y) if self.eye_pos is not None else (x, y)
                else:
                    self.eye_pos = (px * 0.999 + x * 0.001, py * 0.999 + y * 0.001)
                x, y = self.eye_pos
                x, y = clamp(int(round(x)), HALF_CUT_SIZE, frame_cut.shape[1] - HALF_CUT_SIZE), clamp(int(round(y)), HALF_CUT_SIZE, frame_cut.shape[0] - HALF_CUT_SIZE)
                cut = frame_cut[max(y-HALF_CUT_SIZE,0):y+HALF_CUT_SIZE,max(x-HALF_CUT_SIZE,0):x+HALF_CUT_SIZE]
                if dump:
                    cv2.rectangle(frame0,(x - HALF_CUT_SIZE,y - HALF_CUT_SIZE),(x + HALF_CUT_SIZE,y + HALF_CUT_SIZE),(255,255,0),5)
            else:
                cut = frame_cut[:CUT_SIZE, :CUT_SIZE]

        if self.accumu is None:
            self.accumu = f.copy()

        if dump:
            dumps = [frame, self.accumu.astype(np.uint8), frame0]
            # dumps = [frame0]
        self.cnt = self.cnt + 1
        if self.cnt > 5:
            diff = np.abs(f - self.accumu) * 4
            if not self.fixed_eyepos:
                if (x,y) != (0,0):
                    diff = diff[max(y-HALF_CUT_SIZE,0):y+HALF_CUT_SIZE,max(x-HALF_CUT_SIZE,0):x+HALF_CUT_SIZE]
                else:
                    diff = diff[:CUT_SIZE, :CUT_SIZE]
            diff_val = np.sum(diff)
            if dump:
                dumps.append(np.clip(diff, 0, 255).astype(np.uint8))
            blink = diff_val > np.product(diff.shape) * 20
            # print(diff_val / 255.0 / np.product(diff.shape), blink)
            # print('{:7.2F}'.format(diff_val / np.product(diff.shape)))

            # if self.cnt > 30:
            #     blink = diff_val >= self.threshold
            # if not blink:
            #     if self.cnt == 6:
            #         self.threshold = diff_val / 0.2 # 0.7
            #     else:
            #         self.threshold = self.threshold * 0.9 + diff_val / 0.2 * 0.1
            # else:
            #     pass
            #     # print('Blink!')

        if (x,y) == (0,0):
            blink = True    # force rewrite output to throw the image
        if not blink:
            self.accumu *= 0.6
            self.accumu += f * 0.4
            if self.prev_blink > 0:
                self.prev_blink -= 1
                blink = True
        else:
            self.accumu *= 0.8
            self.accumu += f * 0.2
            if self.prev_blink == 0:
                blink = False
            self.prev_blink = 8
        if dump:
            return blink, cut, dumps
        else:
            return blink, cut


import sys

def main():
    fn = 'cut1.mp4'
    if len(sys.argv) == 2:
        fn = sys.argv[1]

    ctx = Analyzer(EYE_POS)
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
        if not ctx.try_load_cache(fn + ('.leye' if USE_LEFT else '.reye')):
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
                    if USE_LEFT:
                        frame = frame[0:720,0:1280]
                        frame = cv2.flip(cv2.transpose(frame), 0)
                    else:
                        frame = frame[0:720,1280:2560]
                        frame = cv2.flip(cv2.transpose(frame), 1)
                    ctx.firstpass(frame, frame_all // int(fps))
            ctx.save_cache(fn + ('.leye' if USE_LEFT else '.reye'))
            print('{:.2%}  '.format(1.))
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)   # reset position

    while True:
        succeeded, frame = video.read()
        if not succeeded:
            break
        if cv2.waitKey(1) == 27:
            break
        if USE_LEFT:
            frame = frame[0:720,0:1280]
            frame = cv2.flip(cv2.transpose(frame), 0)
        else:
            frame = frame[0:720,1280:2560]
            frame = cv2.flip(cv2.transpose(frame), 1)

        blink, cut, dumps = ctx.analyze(frame, dump=True)
#        blink, cut = ctx.analyze(frame, dump=False); dumps=[]
        if blink:
            cv2.rectangle(cut, (0, 0), (cut.shape[0], cut.shape[1]), (0,0,255), 5)

        cv2.imshow('Preview', cut)

        for i, dump in enumerate(dumps):
            if dump is not None:
                cv2.imshow('Dump {}'.format(i), dump)

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
