"""A class that uses various methods to count outliers."""
import cv2
import numpy as np


class USAC:
    """A class for counting inliers."""

    def __init__(
        self,
        method=cv2.USAC_MAGSAC,
        threshold=0.7,
        confidence=0.9999,
        max_iterations=10000,
        sigma=3.0
    ):
        """Init the usac class based on the method used."""
        self.method = method
        self.params = cv2.UsacParams()

        self.params.confidence = confidence
        self.params.maxIterations = max_iterations
        self.params.threshold = threshold
        self.params.loMethod = method

        if method == cv2.USAC_PROSAC:
            self.params.prosacSortedSampling = True

    def est_homography(self, pts_1, pts_2):
        """Estimate the homography given 2 sets of pts."""
        homography, mask = cv2.findHomography(
            pts1, pts2,
            method=self.method,
            ransacReprojThreshold=self.params.threshold,
            maxIters=int(self.params.maxIterations),
            confidence=self.params.confidence,
            usacParams=self.params
        )
        return homography, mask

    def eval_and_est_homography(self, pts_1, pts_2):
        """Compute the quality of the homography."""
        H, mask = self.est_homography(pts1, pts2)
        metrics = {'inliers': 0, 'ratio': 0.0, 'mean_error': float('inf')}

        if H is None or mask is None:
            return H, mask, metrics

        mask = mask.ravel().astype(bool)
        inl1 = pts1[mask]
        inl2 = pts2[mask]
        n_inliers = inl1.shape[0]
        total = pts1.shape[0]

        # Compute reprojection errors for inliers
        if n_inliers > 0:
            # to homogeneous
            h_pts1 = np.hstack([inl1, np.ones((n_inliers, 1), dtype=inl1.dtype)])
            proj = (H @ h_pts1.T).T
            proj = proj[:, :2] / proj[:, 2:3]
            errors = np.linalg.norm(proj - inl2, axis=1)
            mean_err = float(np.mean(errors))
        else:
            mean_err = float('inf')

        metrics['inliers'] = int(n_inliers)
        metrics['ratio'] = n_inliers / float(total)
        metrics['mean_error'] = mean_err

        return H, mask.astype(np.uint8), metrics
