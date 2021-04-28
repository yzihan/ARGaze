#!/usr/bin/env python
import os
import sys
import cv2
import numpy as np
import time
from ARTarget import Analyzer as ARTargetAnalyzer
from CalibTarget import Analyzer as CalibTargetAnalyzer
from BlinkTest import Analyzer as BlinkAnalyzer
from BlinkTest import CUT_SIZE as EyeImageSize

CUT_SIZE=512

VIDEO_MAP={
    'scene': (720,0,1440,1280),
    'leye': (0,0,720,1280),
    'reye': (0,1280,720,2560),
    }

USE_VIDEO=True
SERIAL = 1

pref = '.'

def parse_eyelist(fn: str, serial: int) -> [((int, int), int, int)]:    # return: list of (left-top pos x&y, width, height)
    with open(fn, 'r') as file:
        k = {}
        prev_v = None
        for f in map(lambda x: x.split(':'), file.read().split('\n')):
            if len(f) != 0:
                if len(f) != 2:
                    if f == ['']:
                        continue
                    print('Invalid eyelist.txt format: len(split<":">) != 2, (f =', f, ')')
                    exit(1)
                v = f[1].strip().split(',')
                if v == ['same']:
                    if prev_v is None:
                        print('The first line cannot be "same"')
                        exit(1)
                elif len(v) != 2:
                    print('Invalid eyelist.txt format: len(split<",">) != 2, (v =', v, ')')
                    exit(1)
                else:
                    prev_v = []
                    for iv in v:
                        iv = iv.strip()
                        x, t = iv.split('x')
                        y, w, h = t.split('+')
                        prev_v.append([
                            int(x),
                            int(y),
                            int(w),
                            int(h),
                        ])
                v = prev_v
                k[int(f[0])] = v
    return k[int(serial)]

def load_params() -> (str, str):
    global USE_VIDEO
    global SERIAL

    args = sys.argv
    SERIAL = 1
    if len(args) > 0:
        if len(args) not in [3, 4]:
            print('Usage: {} <prefix> <serial> ["video" | "image"]'.format(args[0]))
            exit(1)
        pref = args[1]
        SERIAL = int(args[2])
        if len(args) > 3:
            form = args[3]
            if form == 'video':
                USE_VIDEO = True
            elif form == 'image':
                USE_VIDEO = False
            else:
                print('Dataset type must be "video" or "image"')
                exit(1)

    fn = os.path.join(pref, 'cut{}.mp4')
    preview = os.path.join(pref, 'cut{}.preview.mp4')
    outpath = os.path.join(pref, 'serial{}')
    listfn = os.path.join(pref, 'eyelist.txt')
    return fn.format(SERIAL), outpath.format(SERIAL), preview.format(SERIAL), listfn.format(SERIAL)

def save_image(fn: str, img: np.ndarray, incremental: bool = True):
    if not (incremental and os.path.isfile(fn)):
        cv2.imwrite(fn, img)

def main():
    fn, outpath, preview, listfn = load_params()
    eyepos = parse_eyelist(listfn, SERIAL)

    EyePos = []
    
    for m,(x,y,w,h) in zip([VIDEO_MAP['leye'],VIDEO_MAP['reye']],eyepos):
        cx, cy = x + w // 2, y + h // 2
        cx = min(max(cx, m[1] + CUT_SIZE // 2), m[3] - CUT_SIZE // 2) - m[1]
        cy = min(max(cy, m[0] + CUT_SIZE // 2), m[2] - CUT_SIZE // 2) - m[0]
        EyePos.append((cx, cy))
    EyePos[0] = (EyePos[0][1], VIDEO_MAP['leye'][3] - VIDEO_MAP['leye'][1] - EyePos[0][0])
    EyePos[1] = (VIDEO_MAP['reye'][2] - VIDEO_MAP['reye'][0] - EyePos[1][1], EyePos[1][0])
    print(EyePos)

    ctx = (ARTargetAnalyzer if SERIAL % 2 == 1 else CalibTargetAnalyzer)()
    blink_ctx_l = BlinkAnalyzer(EyePos[0])
    blink_ctx_r = BlinkAnalyzer(EyePos[1])
    video = cv2.VideoCapture(fn)
    if not video.isOpened():
        print('Video device not opened')
        exit(1)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_all = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('File: {}'.format(fn))
    print('FPS: {}'.format(fps))
    print('Frame Count: {}'.format(frame_all))
    print('Length: {}s'.format(frame_all / fps))

    preview_video = cv2.VideoCapture(preview)
    if not preview_video.isOpened():
        print('Preview Video device not opened')
        exit(1)

    preview_frame_all = int(preview_video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Preview: {}'.format(preview))
    print('Preview Frame Count: {}'.format(preview_frame_all))

    cache1 = 'firstpass' in dir(ctx)
    cache2 = 'firstpass' in dir(blink_ctx_l)
    cache3 = 'firstpass' in dir(blink_ctx_r)
    if cache1 or cache2 or cache3:
        if cache1:
            cache1 = ctx.try_load_cache(preview) # true if cache hit
        else:
            cache1 = True
        if cache2:
            cache2 = blink_ctx_l.try_load_cache(preview + '.leye')
        else:
            cache2 = True
        if cache3:
            cache3 = blink_ctx_r.try_load_cache(preview + '.reye')
        else:
            cache3 = True
        if not (cache1 and cache2 and cache3):
            cnt = preview_frame_all
            while True:
                succeeded, frame = preview_video.read()
                if not succeeded:
                    if cnt > 0:
                        continue
                    else:
                        break
                cnt -= 1

                print('Preprocessing {:.2%}  '.format(preview_video.get(cv2.CAP_PROP_POS_FRAMES) / preview_frame_all), end='\r')

                if not cache1:
                    f = frame[VIDEO_MAP['scene'][0]: VIDEO_MAP['scene'][2], VIDEO_MAP['scene'][1]: VIDEO_MAP['scene'][3]]
                    ctx.firstpass(f, preview_frame_all)
                if not cache2:
                    f = frame[VIDEO_MAP['leye'][0]: VIDEO_MAP['leye'][2], VIDEO_MAP['leye'][1]: VIDEO_MAP['leye'][3]]
                    f = cv2.flip(cv2.transpose(f), 0)
                    blink_ctx_l.firstpass(f, preview_frame_all)
                if not cache3:
                    f = frame[VIDEO_MAP['reye'][0]: VIDEO_MAP['reye'][2], VIDEO_MAP['reye'][1]: VIDEO_MAP['reye'][3]]
                    f = cv2.flip(cv2.transpose(f), 1)
                    blink_ctx_r.firstpass(f, preview_frame_all)
            if not cache1:
                ctx.save_cache(preview)
            if not cache2:
                blink_ctx_l.save_cache(preview + '.leye')
            if not cache3:
                blink_ctx_r.save_cache(preview + '.reye')
            print('Preprocessing {:.2%}  '.format(1.))
            preview_video.release()

    path = [ os.path.join(outpath,str(i)) for i in range(3) ]
    for p in path:
        if not os.path.isdir(p):
            os.makedirs(p)
    if USE_VIDEO:
        video_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#        dim = [(480,640),(640,480),(640,480)]
        dim = [
            (VIDEO_MAP['scene'][3] - VIDEO_MAP['scene'][1], VIDEO_MAP['scene'][2] - VIDEO_MAP['scene'][0]),
            (EyeImageSize, EyeImageSize),
            (EyeImageSize, EyeImageSize),
            ]
        videos = [ cv2.VideoWriter(os.path.join(path[i], 'all.mp4'), video_fourcc, fps, dim[i]) for i in range(3) ]
    else:
        outfn = [ os.path.join(path[i], '{}.png') for i in range(3) ]

    targetpos = []
    serial = 0
    progress = 0

    queue = []
    queue_size = 10

    blink_counter = 0
    blink_counter_reset = 4

    slice_count = 0
    prev_drop = True

    st = time.time()
    ct = time.time()
    prev_progress = 0
    while True:
        succeeded, frame = video.read()
        if not succeeded:
            break
        scenery = frame[VIDEO_MAP['scene'][0]: VIDEO_MAP['scene'][2], VIDEO_MAP['scene'][1]: VIDEO_MAP['scene'][3]]
        leye = frame[VIDEO_MAP['leye'][0]: VIDEO_MAP['leye'][2], VIDEO_MAP['leye'][1]: VIDEO_MAP['leye'][3]]
        leye = cv2.flip(cv2.transpose(leye), 0)
        reye = frame[VIDEO_MAP['reye'][0]: VIDEO_MAP['reye'][2], VIDEO_MAP['reye'][1]: VIDEO_MAP['reye'][3]]
        reye = cv2.flip(cv2.transpose(reye), 1)

        # cv2.imshow('scenery', scenery)
        # cv2.imshow('leye', leye)
        # cv2.imshow('reye', reye)

        blink_l, cut_l = blink_ctx_l.analyze(leye, dump = False)
        blink_r, cut_r = blink_ctx_r.analyze(reye, dump = False)
        blink = blink_l or blink_r
        cnt, (x, y), r = ctx.analyze(scenery, dump=False)

        if cnt == 1:
            queue.append((scenery, cut_l, cut_r, [x, y], blink))
        else:
            queue.append((scenery, cut_l, cut_r, None, blink))

        while len(queue) >= queue_size:
            a,b,c,pos,bl = queue[0]
            _,_,_,_,bl1 = queue[1]
            del queue[0]
            if bl:
                blink_counter = blink_counter_reset
            else:
                blink_counter = blink_counter - 1
            if bl or bl1 or pos is None or blink_counter > 0:
                prev_drop = True
                continue
            if prev_drop:
                slice_count += 1
                prev_drop = False
            if USE_VIDEO:
                videos[0].write(a)
                videos[1].write(b)
                videos[2].write(c)
            else:
                save_image(outfn[0].format(serial), a)
                save_image(outfn[1].format(serial), b)
                save_image(outfn[2].format(serial), c)
            targetpos.append(pos)
            serial = serial + 1
        nt = time.time()
        if nt - ct >= 1:
            rate = (progress - prev_progress) / (nt - ct)
            prev_progress = progress
            ct = nt
            print('Progress: {0}/{1}  Usability: {2:.2%}  Speed: {4:.2F}fps  ETA: {3:.2F}s'.format(progress, frame_all, serial/progress, (frame_all-progress)/progress*(ct-st), rate), end='  \r')
        progress = progress + 1
    
    while len(queue) > 0:
        a,b,c,pos,bl = queue[0]
        bl1 = False
        if len(queue) > 1:
            _,_,_,_,bl1 = queue[1]
        del queue[0]
        if bl:
            blink_counter = blink_counter_reset
        else:
            blink_counter = blink_counter - 1
        if bl or bl1 or pos is None or blink_counter > 0:
            prev_drop = True
            continue
        if prev_drop:
            slice_count += 1
            prev_drop = False
        if USE_VIDEO:
            videos[0].write(a)
            videos[1].write(b)
            videos[2].write(c)
        else:
            save_image(outfn[0].format(serial), a)
            save_image(outfn[1].format(serial), b)
            save_image(outfn[2].format(serial), c)
        targetpos.append(pos)
        serial = serial + 1
    del queue

    ct = time.time()
    if slice_count == 0:
        print('Progress: {}/{}  Usability: {:.2%}  Usable: {}  Speed: {:.2F}fps  Subslice Count: {}  Average Subslice Length: N/A    '.format(progress,frame_all,serial/progress,serial,progress/(ct-st),slice_count))
    else:
        print('Progress: {}/{}  Usability: {:.2%}  Usable: {}  Speed: {:.2F}fps  Subslice Count: {}  Average Subslice Length: {:.1F} frame(s)    '.format(progress,frame_all,serial/progress,serial,progress/(ct-st),slice_count,serial/slice_count))

    np.save(os.path.join(outpath, 'target.npy'), np.array(targetpos))

    if USE_VIDEO:
        for v in videos:
            v.release()

    video.release()
#    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
