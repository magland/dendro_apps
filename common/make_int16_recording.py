import shutil
import os
import numpy as np
import spikeinterface.preprocessing as spre
import spikeinterface as si


def make_int16_recording(recording: si.BaseRecording, *, dirname: str) -> si.BinaryRecordingExtractor:
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    fname = f'{dirname}/recording.dat'
    if recording.get_num_segments() != 1:
        raise NotImplementedError("Can only write recordings with a single segment")
    if recording.get_dtype().kind == 'f':
        # need to scale data so we don't lose precision
        # Look at the first ten seconds of data
        traces0 = recording.get_traces(start_frame=0, end_frame=int(recording.get_sampling_frequency() * 10))
        max_abs_val = np.max(np.abs(traces0))
        median_abs_val = np.median(np.abs(traces0))
        scale_factor = _determine_optimal_scale_factor_for_int16(max_abs_val=max_abs_val, median_abs_val=median_abs_val)
        if scale_factor != 1:
            recording = spre.scale(recording, gain=scale_factor)

    # if recording.get_dtype() != np.int16:
    #     # important so it won't be rewritten for kilosort3
    #     raise NotImplementedError(f"Can only write recordings with dtype int16. This recording has dtype {recording.get_dtype()}")
    si.BinaryRecordingExtractor.write_recording(
        recording=recording,
        file_paths=[fname],
        dtype='int16',
        n_jobs=1, # There may be some issues with parallelization (h5py and remfile, who knows)
        chunk_duration='20s', # this defaults to 1s which is inefficient for download
    )
    ret = si.BinaryRecordingExtractor(
        file_paths=[fname],
        sampling_frequency=recording.get_sampling_frequency(),
        channel_ids=recording.get_channel_ids(),
        num_chan=recording.get_num_channels(),
        dtype='int16'
    )
    ret.set_channel_locations(recording.get_channel_locations())
    return ret

def _determine_optimal_scale_factor_for_int16(*, max_abs_val, median_abs_val) -> float:
    """
    Determine the optimal scale factor for int16 data.
    """
    # We want the median abs val to be between 20 and 100
    # And the max abs val should be less than 5000 (The strict max is 32767, but we are being conservative because we are only examining a small portion of the data)
    scale_factor = 1
    if median_abs_val * scale_factor < 20:
        scale_factor = 20 / median_abs_val
    if median_abs_val * scale_factor > 100:
        scale_factor = 100 / median_abs_val
    if max_abs_val * scale_factor > 5000:
        scale_factor = 5000 / max_abs_val
    return scale_factor