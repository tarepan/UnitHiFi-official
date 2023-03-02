"""©tarepan, licensed to the world under MIT LICENSE."""

import numpy as np
from multiseries import clip_segment, match_length


def test_match_length_1_series():
    """
                  0                 1                   2
                  1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 
    series1(hop2) *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- => 22 unit
    """
    h1 = 2
    s1 = np.random.randint(10, size=22)
    lcm = 2
    n_unit = 22
    np.testing.assert_array_equal(match_length([(s1, h1)]), [s1[..., :n_unit * lcm // h1]])


def test_match_length_2_series():
    """
                  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |     
    series1(hop2) *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- => 7 unit + tail
    series2(hop3) +--+--+--+--+--+--+--+--+--+--               => 5 unit
    """
    h1, h2 = 2, 3
    s1 = np.random.randint(10, size=22)
    s2 = np.random.randint(10, size=10)
    lcm = 6
    n_unit = 5
    assert all([
        np.array_equal(estim, gt) for estim, gt
        in zip(
            match_length([(s1, h1), (s2, h2)]),
            [s1[..., :n_unit*(lcm//h1)], s2[..., :n_unit*(lcm//h2)]],
        )
    ])


def test_match_length_2_series_multidim():
    """
                  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |     
    series1(hop2) *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- => 7 unit + tail
    series2(hop3) +--+--+--+--+--+--+--+--+--+--               => 5 unit
    """
    h1, h2 = 2, 3
    s1 = np.random.randint(10, size=(   4, 22))
    s2 = np.random.randint(10, size=(3, 2, 10))
    lcm = 6
    n_unit = 5
    assert all([
        np.array_equal(estim, gt) for estim, gt
        in zip(
            match_length([(s1, h1), (s2, h2)]),
            [s1[..., :n_unit*(lcm//h1)], s2[..., :n_unit*(lcm//h2)]],
        )
    ])


def test_match_length_2_series_minlength():
    """
                  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |     
    series1(hop2) *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- => 7 unit + tail
    series2(hop3) +--+--+--+--+--+--+--+--+--+--               => 5 unit
    """
    h1, h2 = 2, 3
    s1 = np.random.randint(10, size=22)
    s2 = np.random.randint(10, size=10)
    lcm = 6
    n_unit = 5
    matched_s1_single = s1[..., :n_unit*(lcm//h1)]
    matched_s2_single = s2[..., :n_unit*(lcm//h2)]

    for estim, gt in zip(
            match_length([(s1, h1), (s2, h2)], 70),
            [
                np.concatenate([matched_s1_single, matched_s1_single, matched_s1_single], axis=-1),
                np.concatenate([matched_s2_single, matched_s2_single, matched_s2_single], axis=-1),
            ],
    ):
        assert np.array_equal(estim, gt)


def test_match_length_3_series():
    """
                  |   unit1   |   unit2   |   unit3   | tail
    series1(hop2) *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- => 3 unit + tail
    series2(hop3) +--+--+--+--+--+--+--+--+--+--               => 2 unit + tail
    series3(hop4) ^...^...^...^...^...^...^...^...^...         => 3 unit
    """
    h1, h2, h3 = 2, 3, 4
    s1 = np.random.randint(10, size=22)
    s2 = np.random.randint(10, size=10)
    s3 = np.random.randint(10, size= 9)
    lcm = 12
    n_unit = 2
    assert all([
        np.array_equal(estim, gt) for estim, gt
        in zip(
            match_length([(s1, h1), (s2, h2), (s3, h3)]),
            [s1[..., :n_unit*(lcm//h1)], s2[..., :n_unit*(lcm//h2)], s3[..., :n_unit*(lcm//h3)]],
        )
    ])


def test_match_length_manual():
    """
                  |   unit1   |   unit2   |   unit3   | tail
    series1(hop2) 1-2-3-4-5-4-3-2-1-0-1-2-3-4-5-6-7-8-9-8-7-6- => 3 unit + tail
    series2(hop3) 9--8--7--6--5--6--5--4--3--2--               => 2 unit + tail
    series3(hop4) 0...1...2...3...4...5...4...3...2...         => 3 unit
    """
    h1, h2, h3 = 2, 3, 4
    s1 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6])
    s2 = np.array([9, 8, 7, 6, 5, 6, 5, 4, 3, 2])
    s3 = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2])

    for estim, gt in zip(
            match_length([(s1, h1), (s2, h2), (s3, h3)], 30),
            [
                np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2,    1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2,]),
                np.array([9, 8, 7, 6, 5, 6, 5, 4,                9, 8, 7, 6, 5, 6, 5, 4,]),
                np.array([0, 1, 2, 3, 4, 5,                      0, 1, 2, 3, 4, 5,])
            ],
    ):
        assert np.array_equal(estim, gt)

#####################################################################################################
def test_clip_segment_1_series():
    """
                  0                 1                   2
                  1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 
    series1(hop2) *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- => 22 unit
    """
    h1 = 2
    s1 = np.random.randint(10, size=22)
    sample_per_unit = np.lcm.reduce([h1])
    len_segment = sample_per_unit * 5
    start = sample_per_unit * 1
    assert len_segment % sample_per_unit == 0, "Pre-test condition 'len_segment=N*sample_per_unit' is not satisfied."
    assert       start % sample_per_unit == 0, "Pre-test condition       'start=N*sample_per_unit' is not satisfied."
    
    for estim, gt in zip(
            clip_segment([(s1, h1)], len_segment, start),
            [
                s1[..., (start//sample_per_unit) : (start//sample_per_unit) + (len_segment//sample_per_unit)],
            ],
    ):
        assert np.array_equal(estim, gt)


def test_clip_segment_2_series():
    """
                  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |     
    series1(hop2) *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- => 7 unit + tail
    series2(hop3) +--+--+--+--+--+--+--+--+--+--               => 5 unit
    """
    h1, h2 = 2, 3
    s1 = np.random.randint(10, size=22)
    s2 = np.random.randint(10, size=10)
    sample_per_unit = np.lcm.reduce([h1, h2])
    len_segment = sample_per_unit * 2
    start = sample_per_unit * 2
    assert len_segment % sample_per_unit == 0, "Pre-test condition 'len_segment=N*sample_per_unit' is not satisfied."
    assert       start % sample_per_unit == 0, "Pre-test condition       'start=N*sample_per_unit' is not satisfied."

    for estim, gt in zip(
            clip_segment([(s1, h1), (s2, h2)], len_segment, start),
            [
                s1[..., (start//h1) : (start//h1) + (len_segment//h1)],
                s2[..., (start//h2) : (start//h2) + (len_segment//h2)],
            ],
    ):
        assert np.array_equal(estim, gt)


def test_clip_segment_3_series():
    """
                  |   unit1   |   unit2   |   unit3   
    series1(hop2) *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    series2(hop3) +--+--+--+--+--+--+--+--+--+--+--+--
    series3(hop4) ^...^...^...^...^...^...^...^...^...
    """
    h1, h2, h3 = 2, 3, 4
    s1 = np.random.randint(10, size=18)
    s2 = np.random.randint(10, size=12)
    s3 = np.random.randint(10, size= 9)
    sample_per_unit = np.lcm.reduce([h1, h2, h3])
    len_segment = sample_per_unit * 2
    start       = sample_per_unit * 1
    assert len_segment % sample_per_unit == 0, "Pre-test condition 'len_segment=N*sample_per_unit' is not satisfied."
    assert       start % sample_per_unit == 0, "Pre-test condition       'start=N*sample_per_unit' is not satisfied."
    # len_segment should be n_unit*N
    # start should be n_unit*N
    # In API, clip_random_segment, so start can be auto-calculated
    # len_segment is specified by user
    #
    # Feature-nize
    # Save
    # Clip

    for estim, gt in zip(
            clip_segment([(s1, h1), (s2, h2), (s3, h3)], len_segment, start),
            [
                s1[..., (start//h1) : (start//h1) + (len_segment//h1)],
                s2[..., (start//h2) : (start//h2) + (len_segment//h2)],
                s3[..., (start//h3) : (start//h3) + (len_segment//h3)],
            ],
    ):
        assert np.array_equal(estim, gt)


def test_clip_segment_3_series_mutlidim():
    """
                  |   unit1   |   unit2   |   unit3   
    series1(hop2) *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    series2(hop3) +--+--+--+--+--+--+--+--+--+--+--+--
    series3(hop4) ^...^...^...^...^...^...^...^...^...
    """
    h1, h2, h3 = 2, 3, 4
    s1 = np.random.randint(10, size=(     3, 18))
    s2 = np.random.randint(10, size=(        12))
    s3 = np.random.randint(10, size=( 2,  1,  9))
    sample_per_unit = np.lcm.reduce([h1, h2, h3])
    len_segment = sample_per_unit * 2
    start       = sample_per_unit * 1
    assert len_segment % sample_per_unit == 0, "Pre-test condition 'len_segment=N*sample_per_unit' is not satisfied."
    assert       start % sample_per_unit == 0, "Pre-test condition       'start=N*sample_per_unit' is not satisfied."
    # len_segment should be n_unit*N
    # start should be n_unit*N
    # In API, clip_random_segment, so start can be auto-calculated
    # len_segment is specified by user
    #
    # Feature-nize
    # Save
    # Clip

    for estim, gt in zip(
            clip_segment([(s1, h1), (s2, h2), (s3, h3)], len_segment, start),
            [
                s1[..., (start//h1) : (start//h1) + (len_segment//h1)],
                s2[..., (start//h2) : (start//h2) + (len_segment//h2)],
                s3[..., (start//h3) : (start//h3) + (len_segment//h3)],
            ],
    ):
        assert np.array_equal(estim, gt)


def test_clip_segment_manual():
    """
                  |   unit1   |   unit2   |   unit3   
    series1(hop2) 1-2-3-4-5-4-3-2-1-0-1-2-3-4-5-6-7-8-
    series2(hop3) 9--8--7--6--5--6--5--4--3--2--1--0--
    series3(hop4) 0...1...2...3...4...5...4...3...2...
    """
    h1, h2, h3 = 2, 3, 4
    s1 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8,])
    s2 = np.array([9, 8, 7, 6, 5, 6, 5, 4, 3, 2, 1, 0])
    s3 = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2])
    sample_per_unit = np.lcm.reduce([h1, h2, h3])
    len_segment = sample_per_unit * 2
    start       = sample_per_unit * 1
    assert len_segment % sample_per_unit == 0, "Pre-test condition 'len_segment=N*sample_per_unit' is not satisfied."
    assert       start % sample_per_unit == 0, "Pre-test condition       'start=N*sample_per_unit' is not satisfied."

    for estim, gt in zip(
            clip_segment([(s1, h1), (s2, h2), (s3, h3)], len_segment, start),
            [
                np.array([3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8,]),
                np.array([5, 6, 5, 4, 3, 2, 1, 0]),
                np.array([3, 4, 5, 4, 3, 2]),
            ],
    ):
        assert np.array_equal(estim, gt)

