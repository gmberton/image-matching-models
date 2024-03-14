from sklearn.linear_model import LogisticRegression 
import numpy as np
import os

from matching import lightglue, loftr, roma, dedode, handcrafted, duster, se2loftr, steerers, matching_toolbox


matchers = {
    'loftr': loftr.LoftrMatcher,
    'se2loftr': se2loftr.Se2LoFTRMatcher,
    'sift-lg': lightglue.SiftLightGlue,
    'superpoint-lg': lightglue.SuperpoingLightGlue,
    'disk-lg': lightglue.DiskLightGlue,
    'aliked-lg': lightglue.AlikedLightGlue,
    'doghardnet-lg': lightglue.DognetLightGlue,
    'roma': roma.RomaMatcher,
    'dedode': dedode.DedodeMatcher,
    'steerers':steerers.SteererMatcher,
    'sift-nn': handcrafted.SiftNNMatcher,
    'orb-nn': handcrafted.OrbNNMatcher,
    'patch2pix': matching_toolbox.Patch2pixMatcher,
    'patch2pix_superglue': matching_toolbox.SuperGluePatch2pixMatcher,
    'superglue': matching_toolbox.SuperGlueMatcher,
    'r2d2': matching_toolbox.R2D2Matcher,
    'd2net': matching_toolbox.D2netMatcher,
    'duster': duster.DusterMatcher,
    'doghardnet-nn': matching_toolbox.DogAffHardNNMatcher
}


def get_matcher(matcher_name="sift-lg", device="cpu", max_num_keypoints=2048, *args, **kwargs):
    assert matcher_name in matchers

    matcher_class = matchers[matcher_name]
    matcher = matcher_class(device, max_num_keypoints, *args, **kwargs)
    
    return matcher


def compute_threshold(true_matches, false_matches, thresh=0.95):
    assert isinstance(true_matches, list)
    assert isinstance(true_matches, list)

    if (len(true_matches) < 4):
        return 4
    # logistic_model = lambda x: 1 / (1 + np.exp(-x))
    
    X_r = np.array(true_matches).reshape(-1, 1)
    X_w = np.array(false_matches).reshape(-1, 1)
    X = np.concatenate((X_r, X_w))
    
    Y_r = np.ones(len(true_matches), dtype=int)
    Y_w = np.zeros(len(false_matches), dtype=int)
    Y = np.concatenate((Y_r, Y_w))
    
    lr = LogisticRegression()
    lr.fit(X, Y)

    f_y = - np.log((1-thresh)/thresh)
    match_thresh = (f_y - lr.intercept_)/lr.coef_
    return match_thresh.item()