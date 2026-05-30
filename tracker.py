import numpy as np
import cv2
import scipy.linalg
from scipy.optimize import linear_sum_assignment

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    State representation: [x_c, y_c, a, h, vx, vy, va, vh]
    Measurement representation: [x_c, y_c, a, h]
    """
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.dot(self._motion_mat, np.dot(covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.dot(self._update_mat, np.dot(covariance, self._update_mat.T)) + innovation_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.dot(kalman_gain, np.dot(projected_cov, kalman_gain.T))
        return new_mean, new_covariance


def warp_box(mean, H):
    """
    Warp bounding box [xc, yc, a, h] using homography matrix H.
    """
    xc, yc, a, h = mean[0:4]
    w = a * h
    # Define corners: top-left, top-right, bottom-left, bottom-right
    corners = np.array([
        [xc - w/2, yc - h/2],
        [xc + w/2, yc - h/2],
        [xc - w/2, yc + h/2],
        [xc + w/2, yc + h/2]
    ])
    # Homogeneous coordinates
    corners_hom = np.hstack([corners, np.ones((4, 1))])
    # Project corners
    warped_hom = np.dot(H, corners_hom.T).T
    warped = warped_hom[:, :2] / warped_hom[:, 2:3]
    
    # Bounding box of the warped corners
    x1, y1 = np.min(warped, axis=0)
    x2, y2 = np.max(warped, axis=0)
    
    new_w = x2 - x1
    new_h = y2 - y1
    new_xc = x1 + new_w / 2
    new_yc = y1 + new_h / 2
    new_a = new_w / new_h if new_h > 0 else a
    
    return np.array([new_xc, new_yc, new_a, new_h])


def warp_state(mean, H):
    """
    Warp state [xc, yc, a, h, vx, vy, va, vh] using homography matrix H.
    """
    xc, yc, a, h, vx, vy, va, vh = mean
    new_box = warp_box(mean, H)
    new_xc, new_yc, new_a, new_h = new_box
    
    # Project velocity vector [vx, vy] by taking differences of warped points
    pt1_hom = np.array([xc, yc, 1.0])
    pt2_hom = np.array([xc + vx, yc + vy, 1.0])
    
    warped1_hom = np.dot(H, pt1_hom)
    warped1 = warped1_hom[:2] / warped1_hom[2]
    
    warped2_hom = np.dot(H, pt2_hom)
    warped2 = warped2_hom[:2] / warped2_hom[2]
    
    new_vx, new_vy = warped2 - warped1
    
    return np.array([new_xc, new_yc, new_a, new_h, new_vx, new_vy, va, vh])


def warp_covariance(covariance, H):
    """
    Transform state covariance matrix using linear part of homography H.
    """
    A = H[0:2, 0:2]
    M = np.eye(8)
    M[0:2, 0:2] = A
    M[4:6, 4:6] = A
    return np.dot(M, np.dot(covariance, M.T))


class GlobalMotionCompensation:
    """
    Estimates global camera motion between frames using ORB keypoint matching
    and filters out foreground objects to keep tracking robust.
    """
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_frame = None
        self.prev_kps = None
        self.prev_descs = None

    def reset(self):
        self.prev_frame = None
        self.prev_kps = None
        self.prev_descs = None

    def estimate_motion(self, frame, detections=None):
        h_orig, w_orig = frame.shape[:2]
        w_small = 640
        scale = w_orig / w_small
        h_small = int(h_orig / scale)
        
        # Resize to speed up ORB feature matching (~2-5ms vs ~25ms)
        small_frame = cv2.resize(frame, (w_small, h_small))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        small_detections = None
        if detections is not None:
            small_detections = []
            for det in detections:
                x1, y1, x2, y2 = det[:4]
                small_detections.append([x1 / scale, y1 / scale, x2 / scale, y2 / scale])
        
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_kps, self.prev_descs = self.detector.detectAndCompute(gray, None)
            if small_detections is not None and len(small_detections) > 0 and self.prev_kps is not None:
                self.prev_kps, self.prev_descs = self._filter_keypoints(self.prev_kps, self.prev_descs, small_detections)
            return np.eye(3)
            
        kps, descs = self.detector.detectAndCompute(gray, None)
        if small_detections is not None and len(small_detections) > 0 and kps is not None:
            kps, descs = self._filter_keypoints(kps, descs, small_detections)
            
        if self.prev_descs is None or descs is None or len(self.prev_descs) == 0 or len(descs) == 0:
            self.prev_frame = gray
            self.prev_kps, self.prev_descs = kps, descs
            return np.eye(3)
            
        matches = self.matcher.match(self.prev_descs, descs)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:500]
        
        if len(good_matches) < 4:
            self.prev_frame = gray
            self.prev_kps, self.prev_descs = kps, descs
            return np.eye(3)
            
        pts1 = np.float32([self.prev_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H_small, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        
        if H_small is None:
            H_orig = np.eye(3)
        else:
            T = np.array([
                [scale, 0.0, 0.0],
                [0.0, scale, 0.0],
                [0.0, 0.0, 1.0]
            ])
            T_inv = np.array([
                [1.0/scale, 0.0, 0.0],
                [0.0, 1.0/scale, 0.0],
                [0.0, 0.0, 1.0]
            ])
            H_orig = np.dot(T, np.dot(H_small, T_inv))
            
        self.prev_frame = gray
        self.prev_kps, self.prev_descs = kps, descs
        
        return H_orig

    def _filter_keypoints(self, kps, descs, detections):
        if descs is None or len(kps) == 0:
            return kps, descs
            
        mask = np.ones(len(kps), dtype=bool)
        for i, kp in enumerate(kps):
            x, y = kp.pt
            for det in detections:
                x1, y1, x2, y2 = det[:4]
                pad = 5.0
                if (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad):
                    mask[i] = False
                    break
        
        filtered_kps = [k for i, k in enumerate(kps) if mask[i]]
        filtered_descs = descs[mask] if len(filtered_kps) > 0 else None
        
        return filtered_kps, filtered_descs


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """
    Representation of a single track.
    """
    def __init__(self, tlwh, score, class_id):
        # convert to xc, yc, a, h
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        
        self.state = TrackState.New
        self.is_activated = False
        self.score = score
        self.class_id = class_id
        self.track_id = 0
        
        self.history = []  # List of centroids (x_c, y_c) for tail rendering
        self.max_history_len = 30
        
        self.frame_id = 0
        self.start_frame = 0

    @property
    def tlbr(self):
        ret = self._tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # w = a * h
        ret[:2] -= ret[2:] / 2  # x1 = xc - w/2
        return ret

    def to_xyah(self):
        ret = self._tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3] if ret[3] > 0 else 1
        return ret

    def activate(self, kalman_filter, frame_id, track_id):
        self.kalman_filter = kalman_filter
        self.track_id = track_id
        self.mean, self.covariance = self.kalman_filter.initiate(self.to_xyah())
        
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self._add_to_history()

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.to_xyah()
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        if new_id:
            self.track_id = new_track.track_id
        self._add_to_history()

    def update(self, new_track, frame_id):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.to_xyah()
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self._add_to_history()

    def predict(self):
        mean = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean[4] = 0
            mean[5] = 0
            mean[6] = 0
            mean[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean, self.covariance)

    def camera_compensate(self, H):
        """
        Compensate track state and covariance for camera ego-motion.
        """
        if self.mean is not None:
            self.mean = warp_state(self.mean, H)
            self.covariance = warp_covariance(self.covariance, H)

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def _add_to_history(self):
        # Calculate centroid
        tlwh = self.tlwh
        xc = tlwh[0] + tlwh[2] / 2
        yc = tlwh[1] + tlwh[3] / 2
        self.history.append((int(xc), int(yc)))
        if len(self.history) > self.max_history_len:
            self.history.pop(0)


class GMCByteTracker:
    """
    Lightweight tracker integrating Global Motion Compensation and ByteTrack.
    """
    def __init__(self, track_thresh=0.25, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.kalman_filter = KalmanFilter()
        self.gmc = GlobalMotionCompensation()
        
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []     # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.frame_id = 0
        self.track_id_counter = 0

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.track_id_counter = 0
        self.gmc.reset()

    def update(self, output_results, img):
        """
        Updates the tracker with detections from the current frame.
        output_results: list of [x1, y1, x2, y2, score, class_id]
        img: current RGB/BGR frame
        """
        self.frame_id += 1
        
        # 1. Split detections into high-conf and low-conf
        detections = []
        detections_second = []
        
        for det in output_results:
            bbox = det[:4]  # x1, y1, x2, y2
            score = det[4]
            class_id = int(det[5])
            
            # Convert to tlwh (top-left x, y, width, height)
            tlwh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            track = STrack(tlwh, score, class_id)
            
            if score >= self.track_thresh:
                detections.append(track)
            else:
                detections_second.append(track)
                
        # 2. Extract backgrounds and estimate homography
        # We pass detections to mask out keypoints on moving objects
        H = self.gmc.estimate_motion(img, output_results)
        
        # Apply motion compensation to currently tracked and lost tracks
        for track in self.tracked_stracks:
            track.camera_compensate(H)
        for track in self.lost_stracks:
            track.camera_compensate(H)
            
        # 3. Predict track positions for the current frame
        for track in self.tracked_stracks:
            track.predict()
        for track in self.lost_stracks:
            track.predict()
            
        # Unconfirmed tracks (tracks that were just created but not yet confirmed)
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
                
        # Combine tracked and lost tracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # 4. First Association: Match high-confidence detections with the track pool
        dists = iou_distance(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
        
        activated_stracks = []
        refind_stracks = []
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                
        # 5. Second Association: Match low-confidence detections with remaining tracks
        # These are tracks that didn't match with high-confidence detections
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists_second = iou_distance(r_tracked_stracks, detections_second)
        matches_second, u_track_second, u_detection_second = linear_assignment(dists_second, thresh=0.5)
        
        for itracked, idet in matches_second:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            track.update(det, self.frame_id)
            activated_stracks.append(track)
            
        # Any track that remains unassociated is marked lost
        for it in u_track_second:
            track = r_tracked_stracks[it]
            if track.state == TrackState.Tracked:
                track.mark_lost()
                
        # 6. Third Association: Match unconfirmed tracks with remaining high-confidence detections
        # This prevents starting spurious tracks from false detections
        detections_unmatched = [detections[i] for i in u_detection]
        dists_unconfirmed = iou_distance(unconfirmed, detections_unmatched)
        matches_unconfirmed, u_unconfirmed, u_detection_third = linear_assignment(dists_unconfirmed, thresh=0.7)
        
        for itracked, idet in matches_unconfirmed:
            track = unconfirmed[itracked]
            det = detections_unmatched[idet]
            track.update(det, self.frame_id)
            activated_stracks.append(track)
            
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            
        # 7. Start new tracks for remaining unassociated high-confidence detections
        for idet in u_detection_third:
            det = detections_unmatched[idet]
            if det.score >= 0.5:  # Only start tracks for solid detections
                self.track_id_counter += 1
                det.activate(self.kalman_filter, self.frame_id, self.track_id_counter)
                activated_stracks.append(det)
                
        # 8. Manage Lost and Removed tracks
        # Remove tracks that have been lost for too long
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                self.removed_stracks.append(track)
                
        # Update tracker lists
        self.tracked_stracks = [t for t in activated_stracks if t.state == TrackState.Tracked]
        self.lost_stracks = [t for t in joint_stracks(self.lost_stracks, r_tracked_stracks) if t.state == TrackState.Lost]
        self.lost_stracks = [t for t in self.lost_stracks if t not in self.tracked_stracks]
        self.lost_stracks = [t for t in self.lost_stracks if t not in self.removed_stracks]
        self.tracked_stracks = sub_stracks(self.tracked_stracks, self.removed_stracks)
        
        # Return only active confirmed tracks
        return [t for t in self.tracked_stracks if t.is_activated]


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = True
        res.append(t)
    for t in tlistb:
        if t.track_id not in exists:
            exists[t.track_id] = True
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    # Remove tracks in tlistb from tlista
    exists = {t.track_id: True for t in tlistb}
    return [t for t in tlista if t.track_id not in exists]


def iou_distance(atracks, btracks):
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)))
        
    atlbrs = [track.tlbr for track in atracks]
    btlbrs = [track.tlbr for track in btracks]
    
    ious = np.zeros((len(atracks), len(btracks)))
    for i, a in enumerate(atlbrs):
        for j, b in enumerate(btlbrs):
            ious[i, j] = bbox_iou(a, b)
            
    return 1.0 - ious


def bbox_iou(box1, box2):
    # x1, y1, x2, y2
    lt = np.maximum(box1[:2], box2[:2])
    rb = np.minimum(box1[2:], box2[2:])
    wh = np.maximum(0, rb - lt)
    intersection = wh[0] * wh[1]
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0
    return intersection / union


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        
    matches = []
    x, y = linear_sum_assignment(cost_matrix)
    
    unmatched_a = []
    unmatched_b = []
    
    # filter matches based on threshold
    for i, j in zip(x, y):
        if cost_matrix[i, j] > thresh:
            unmatched_a.append(i)
            unmatched_b.append(j)
        else:
            matches.append([i, j])
            
    for i in range(cost_matrix.shape[0]):
        if i not in x:
            unmatched_a.append(i)
            
    for j in range(cost_matrix.shape[1]):
        if j not in y:
            unmatched_b.append(j)
            
    return np.array(matches, dtype=int), unmatched_a, unmatched_b
