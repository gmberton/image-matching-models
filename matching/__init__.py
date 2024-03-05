from matching import lightglue, loftr, roma


matchers = {
    "loftr": loftr.LoftrMatcher,
    "sift-lg": lightglue.SiftLightGlue,
    "superpoint-lg": lightglue.SuperpoingLightGlue,
    "disk-lg": lightglue.DiskLightGlue,
    "aliked-lg": lightglue.AlikedLightGlue,
    "doghardnet-lg": lightglue.DognetLightGlue,
    "roma": roma.RomaMatcher
}


def get_matcher(matcher_name="sift-lg", device="cpu", max_num_keypoints=2048):
    assert matcher_name in matchers

    matcher_class = matchers[matcher_name]
    matcher = matcher_class(device, max_num_keypoints)
    
    return matcher
