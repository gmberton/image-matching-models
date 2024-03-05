from matching import lightglue, loftr, roma, dedode, handcrafted


matchers = {
    'loftr': loftr.LoftrMatcher,
    'sift-lg': lightglue.SiftLightGlue,
    'superpoint-lg': lightglue.SuperpoingLightGlue,
    'disk-lg': lightglue.DiskLightGlue,
    'aliked-lg': lightglue.AlikedLightGlue,
    'doghardnet-lg': lightglue.DognetLightGlue,
    'roma': roma.RomaMatcher,
    'dedode': dedode.DedodeMatcher,
    'sift-nn': handcrafted.SiftNNMatcher,
    'orb-nn': handcrafted.OrbNNMatcher,
}


def get_matcher(matcher_name="sift-lg", device="cpu", max_num_keypoints=2048, *args, **kwargs):
    assert matcher_name in matchers

    matcher_class = matchers[matcher_name]
    matcher = matcher_class(device, max_num_keypoints, *args, **kwargs)
    
    return matcher
