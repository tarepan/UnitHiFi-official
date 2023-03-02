"""©tarepan, licensed to the world under MIT LICENSE."""

import numpy as np
import random


def match_length(serieses_hops, min_length: int = 1):
    """Match lengths of different-hop series.

                  |   unit1   |   unit2   |   unit+   | tail
    series1(hop2) *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- => 3 unit + tail
    series2(hop3) +--+--+--+--+--+--+--+--+--+--               => 2 unit + tail
    series3(hop4) ^...^...^...^...^...^...^...^...^...         => 3 unit

    sample_per_unit = least_common_multiple(series1, series2, series3) = lcm(2,3,4) = 12
    Matched length is just 2 unit.

    Args:
        serieses_hops :: List[(NDArray[(*, T)], int)] - List of series with hop size. All hop sizes should be relative to common hop=1
        min_length - Minimum length [sample]
    Returns:
        :: List[NDArray[(*, T)]] - List of length-matched series
    """

    serieses = list(map(lambda series_hop: series_hop[0], serieses_hops))
    hops     = list(map(lambda series_hop: series_hop[1], serieses_hops))

    # Inter-series length matching
    sample_per_unit = np.lcm.reduce(hops)
    frame_per_units = list(map(lambda hop: sample_per_unit // hop, hops))
    n_unit = min(map(lambda series, frame_per_unit: series.shape[-1] // frame_per_unit, serieses, frame_per_units))
    matched_serieses = list(map(lambda series, frame_per_unit: series[..., : n_unit * frame_per_unit],  serieses, frame_per_units))

    # Minumum length matching
    # len_series [frame] * hop [sample/frame] = len_series [sample]
    len_matched_series = matched_serieses[0].shape[-1] * hops[0]
    if len_matched_series < min_length:
        n_repeat = 1 + (min_length // len_matched_series)
        if n_repeat < 100:
            matched_serieses = list(map(lambda series: np.concatenate([series for _ in range(n_repeat)], axis=-1), matched_serieses))
        else:
            raise RuntimeError(f"Repeat series until min_length={min_length} over 100 times. Something seems to be wrong.")

    return matched_serieses


def clip_segment(serieses_hops, len_segment: int, start: int):
    """clip serieses into segments.
    
    Args:
        serieses_hops :: List[((..., T), int)] - target serieses with hop size
        len_segment - Length of segment (sample scale), should be 'N * lcm(serieses)'
        start - segment start position (sample scale), should be 'M * lcm(serieses)'
    """

    serieses = list(map(lambda series_hop: series_hop[0], serieses_hops))
    hops     = list(map(lambda series_hop: series_hop[1], serieses_hops))

    # start [sample] / hop [sample/frame] = start_frame [frame]
    # len_segment [sample] / hop [sample/frame] = len_frame [frame]
    return list(map(lambda series, hop: series[..., start // hop : (start // hop) + (len_segment // hop)], serieses, hops))


def clip_segment_random(serieses_hops, len_segment: int):
    """Clip serieses into segments with random start point.

    Args:
        serieses_hops :: List[((..., T), int)] - target serieses with hop size, should be length-matched
        len_segment - Length of segment (sample scale), should be 'N * lcm(serieses)'
    """
    serieses = list(map(lambda series_hop: series_hop[0], serieses_hops))
    hops     = list(map(lambda series_hop: series_hop[1], serieses_hops))

    # len_series [frame] * hop [sample/frame] - len_segment [sample]
    start = random.randint(0, serieses[0].shape(-1) * hops[0] - len_segment)

    return clip_segment(serieses_hops, len_segment, start)
