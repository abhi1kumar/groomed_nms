
import numpy as np
import torch


def navneeth_soft_nms(boxes, sigma=0.5, Nt=0.4, threshold=0.001, method=0, shift= 1):
    """
    Taken from
    https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx

    An extra shift is added to accomodate
    """
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0
    N_init = boxes.shape[0]
    keep_orig = np.arange(N_init)

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        # Swap the two indices
        temp = keep_orig[i]
        keep_orig[i] = keep_orig[maxpos]
        keep_orig[maxpos] = temp

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + shift) * (y2 - y1 + shift)
            iw = (min(tx2, x2) - max(tx1, x1) + shift)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + shift)
                if ih > 0:
                    ua = float((tx2 - tx1 + shift) * (ty2 - ty1 + shift) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]

                    # if box score falls below threshold, discard the box by swapping with last box
		            # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        # print("Discarding pos {} in {} length".format(keep_orig[pos], N))
                        # Swap indices
                        temp = keep_orig[N-1]
                        keep_orig[N-1] = keep_orig[pos]
                        keep_orig[pos] = temp
                        # print(keep_orig)
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]

    return keep_orig[:N]


def girshick_nms(dets, thresh, shift= 1):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + shift) * (y2 - y1 + shift)
    order = scores.argsort()[::-1]
    keep = []
    keep_orig = []
    N_dropped = 0
    while order.size > 0:
        i = order[0]
        keep.append(i)
        keep_orig.append(i+N_dropped)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + shift)
        h = np.maximum(0.0, yy2 - yy1 + shift)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
        N_dropped = order.shape[0] - inds.shape[0]

    return keep_orig
