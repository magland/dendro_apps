#!/usr/bin/env python3

import os
import shutil
from typing import List, TYPE_CHECKING
import numpy as np
from dendro.sdk import App, ProcessorBase, BaseModel, Field, InputFile, OutputFile

if TYPE_CHECKING:
    import spikeinterface as si


app = App(
    'spike_sorting_analysis',
    description="Create a spike sorting analysis .h5 file.",
    app_image="ghcr.io/magland/dendro_spike_sorting_analysis:latest",
    app_executable="/app/main.py"
)

class CreateSpikeSortingAnalysisContext(BaseModel):
    recording: InputFile = Field(description='recording .nwb file')
    sorting: InputFile = Field(description='sorting .nwb file')
    output: OutputFile = Field(description='output .h5 file')
    electrical_series_path: str = Field(description='Path to the electrical series in the recording NWB file, e.g., /acquisition/ElectricalSeries')

class CreateSpikeSortingAnalysisProcessor(ProcessorBase):
    name = 'create_spike_sorting_analysis'
    description = 'Create a spike sorting analysis .nh5 file.'
    label = 'Spike sorting analysis'
    tags = ['spike_sorting', 'spike_sorting_analysis']
    attributes = {'wip': True}
    @staticmethod
    def run(context: CreateSpikeSortingAnalysisContext):
        import numpy as np
        import remfile
        from common.NwbRecording import NwbRecording
        from common.NwbSorting import NwbSorting
        from helpers.compute_correlogram_data import compute_correlogram_data
        import spikeinterface.preprocessing as spre
        import h5py
        import simplejson
        from h5_to_nh5 import h5_to_nh5

        snippet_T1 = 30
        snippet_T2 = 30
        chunk_size_sec = 1

        print('Starting spike_sorting_analysis')
        recording_nwb_url = context.recording.get_url()
        sorting_nwb_url = context.sorting.get_url()
        print(f'Input recording NWB URL: {recording_nwb_url}')
        print(f'Input sorting NWB URL: {sorting_nwb_url}')

        # open the remote file
        print('Opening remote input recording file')
        recording_remf = remfile.File(recording_nwb_url)

        print('Creating input recording')
        nwb_recording = NwbRecording(
            file=recording_remf,
            electrical_series_path=context.electrical_series_path
        )

        print('Opening remote input sorting file')
        sorting_remf = remfile.File(sorting_nwb_url)
        nwb_sorting = NwbSorting(sorting_remf)

        print('Preparing filtered recording')
        freq_min = 300
        freq_max = 6000
        recording_filtered = spre.bandpass_filter(nwb_recording, freq_min=freq_min, freq_max=freq_max)

        print('Opening output file')
        output_fname = 'output.h5'
        output_h5 = h5py.File(output_fname, 'w')

        print('Storing unit IDs')
        unit_ids = nwb_sorting.get_unit_ids()
        output_h5.attrs['unit_ids'] = simplejson.dumps([_format_unit_id(unit_id) for unit_id in unit_ids], ignore_nan=True)
        print(f'Unit IDs: {unit_ids}')

        print('Stroring channel locations')
        channel_locations = nwb_recording.get_channel_locations()
        channel_locations = [
            [float(channel_locations[i, j]) for j in range(channel_locations.shape[1])]
            for i in range(channel_locations.shape[0])
        ]
        output_h5.attrs['channel_locations'] = simplejson.dumps(channel_locations, ignore_nan=True)

        print('Storing channel IDs')
        channel_ids = nwb_recording.get_channel_ids()
        output_h5.attrs['channel_ids'] = simplejson.dumps([_format_unit_id(channel_id) for channel_id in channel_ids], ignore_nan=True)

        print('Computing autocorrelograms')
        # create autocorrelograms group
        autocorrelograms_group = output_h5.create_group('autocorrelograms')
        first = True
        all_bin_counts = []
        for unit_id in nwb_sorting.get_unit_ids():
            a = compute_correlogram_data(sorting=nwb_sorting, unit_id1=unit_id, unit_id2=None, window_size_msec=50, bin_size_msec=1)
            bin_edges_sec = a['bin_edges_sec']
            bin_counts = a['bin_counts']
            assert len(bin_counts) == len(bin_edges_sec) - 1
            if first:
                autocorrelograms_group.create_dataset('bin_edges_sec', data=bin_edges_sec)
                first = False
            all_bin_counts.append(bin_counts)
        bin_counts_array = np.zeros((len(unit_ids), len(bin_counts)))
        for ii in range(len(unit_ids)):
            bin_counts_array[ii, :] = all_bin_counts[ii]
        autocorrelograms_group.create_dataset('bin_counts', data=bin_counts_array)

        segment_start_frames, segment_end_frames = _create_subsampling_plan(
            recording_num_frames=recording_filtered.get_num_frames(),
            segment_num_frames=int(recording_filtered.get_sampling_frequency() * 1),
            total_subsampled_num_frames=int(recording_filtered.get_sampling_frequency() * 300)
        )

        recording_cache_dir = 'recording_cache'
        os.mkdir(recording_cache_dir)

        all_average_waveforms: List[np.ndarray] = []
        all_neighborhood_channel_indices: List[List[int]] = []
        all_unit_locations: List[List[float]] = []

        try:
            units_group = output_h5.create_group('units')
            for ii, unit_id in enumerate(unit_ids):
                print('')
                print('=============================================')
                print(f'Processing unit {unit_id} ({ii + 1} of {len(unit_ids)})')
                unit_group = units_group.create_group(f'unit_{unit_id}')
                spike_train_frames = nwb_sorting.get_unit_spike_train(unit_id=unit_id)
                print(f'Number of spikes: {len(spike_train_frames)}')
                unit_group.create_dataset('spike_train', data=spike_train_frames / nwb_recording.get_sampling_frequency())
                print('Subsampling spike train')
                subsampled_spike_train_frames = _subsample_spike_train(
                    spike_train=spike_train_frames,
                    segment_start_frames=segment_start_frames,
                    segment_end_frames=segment_end_frames,
                    margin=snippet_T1 + snippet_T2
                )
                unit_group.create_dataset('subsampled_spike_train', data=subsampled_spike_train_frames / recording_filtered.get_sampling_frequency())
                print('Extracting snippets')
                subsampled_snippets_all_channels = _extract_snippets(
                    recording=recording_filtered,
                    spike_train=subsampled_spike_train_frames,
                    snippet_T1=snippet_T1,
                    snippet_T2=snippet_T2,
                    recording_cache_dir=recording_cache_dir,
                    chunk_size_sec=chunk_size_sec
                )
                print('Computing average waveform')
                average_waveform_all_channels = np.median(subsampled_snippets_all_channels, axis=0)
                unit_group.create_dataset('average_waveform_all_channels', data=average_waveform_all_channels)
                print('Computing unit location')
                unit_location = _compute_unit_location(average_waveform=average_waveform_all_channels, channel_locations=channel_locations)
                all_unit_locations.append(unit_location)
                unit_group.attrs['unit_location'] = simplejson.dumps(unit_location, ignore_nan=True)
                waveform_peak_channel_index = np.argmax(np.max(np.abs(average_waveform_all_channels), axis=0))
                waveform_peak_time_index = np.argmax(np.abs(average_waveform_all_channels[:, waveform_peak_channel_index]))
                print(f'Peak channel index: {waveform_peak_channel_index}')
                print(f'Peak time index: {waveform_peak_time_index}')
                unit_group.attrs['waveform_peak_channel_index'] = int(waveform_peak_channel_index)
                unit_group.attrs['waveform_peak_time_index'] = int(waveform_peak_time_index)
                neighborhood_channel_indices = _get_neighborhood_channel_indices(
                    recording=recording_filtered,
                    peak_channel_index=waveform_peak_channel_index,
                    max_num_channels_per_neighborhood=10
                )
                print(f'Channel neighborhood indices: {neighborhood_channel_indices}')
                unit_group.attrs['neighborhood_channel_indices'] = simplejson.dumps(neighborhood_channel_indices, ignore_nan=True)
                all_neighborhood_channel_indices.append(neighborhood_channel_indices)
                print('Storing snippets')
                subsampled_snippets = subsampled_snippets_all_channels[:, :, neighborhood_channel_indices]
                unit_group.create_dataset('subsampled_snippets', data=subsampled_snippets)
                print('Storing average waveform')
                average_waveform = np.median(subsampled_snippets, axis=0)
                all_average_waveforms.append(average_waveform)
                unit_group.create_dataset('average_waveform', data=average_waveform)
                print('Computing subsampled spike amplitudes')
                subsampled_spike_amplitudes = subsampled_snippets_all_channels[:, waveform_peak_time_index, waveform_peak_channel_index]
                unit_group.create_dataset('subsampled_spike_amplitudes', data=subsampled_spike_amplitudes)

            print('Storing all neighborhood channel indices')
            output_h5.attrs['neighborhood_channel_indices'] = simplejson.dumps(all_neighborhood_channel_indices, ignore_nan=True)

            print('Storing all average waveforms')
            max_neighborhood_size = int(np.max([all_average_waveforms[ii].shape[1] for ii in range(len(all_average_waveforms))]))
            average_waveforms = np.zeros((len(all_average_waveforms), snippet_T1 + snippet_T2, max_neighborhood_size))
            for ii in range(len(all_average_waveforms)):
                average_waveforms[ii, :, :all_average_waveforms[ii].shape[1]] = all_average_waveforms[ii]
            output_h5.create_dataset('average_waveforms', data=average_waveforms)

            print('Storing all unit locations')
            output_h5.attrs['unit_locations'] = simplejson.dumps(all_unit_locations, ignore_nan=True)

            # close output file
            output_h5.close()

            print('Converting to nh5')
            output_fname_nh5 = 'output.nh5'
            h5_to_nh5(output_fname, output_fname_nh5)

            print('Uploading output file')
            context.output.upload(output_fname_nh5)
        finally:
            # remove the recording cache directory
            shutil.rmtree(recording_cache_dir)

def _create_subsampling_plan(*, recording_num_frames: int, segment_num_frames: int, total_subsampled_num_frames: int):
    n = int(np.floor(recording_num_frames / segment_num_frames))
    num_segments = int(np.floor(total_subsampled_num_frames / segment_num_frames))
    segments_to_use = np.zeros((n,), dtype=np.int32)
    if num_segments < n:
        jj = 0
        while jj < n:
            segments_to_use[jj] = 1
            jj += int(np.floor(n / num_segments))
    else:
        segments_to_use[:] = 1
    segment_start_frames = []
    segment_end_frames = []
    for i in range(n):
        if segments_to_use[i] == 1:
            segment_start_frames.append(i * segment_num_frames)
            segment_end_frames.append((i + 1) * segment_num_frames)
    return segment_start_frames, segment_end_frames

def _subsample_spike_train(*, spike_train: np.ndarray, segment_start_frames: List[int], segment_end_frames: List[int], margin: int):
    sub_train_list: List[np.ndarray] = []
    for i in range(len(segment_start_frames)):
        inds = np.where((spike_train >= segment_start_frames[i] + margin) & (spike_train < segment_end_frames[i] - margin))[0]
        sub_train = spike_train[inds]
        sub_train_list.append(sub_train)
    return np.concatenate(sub_train_list)

def _extract_snippets(*,
    recording: 'si.BaseRecording',
    spike_train: np.ndarray,
    snippet_T1: int,
    snippet_T2: int,
    recording_cache_dir: str,
    chunk_size_sec: float
):
    L = len(spike_train)
    T = snippet_T1 + snippet_T2
    M = recording.get_num_channels()
    ret = np.zeros((L, T, M))
    for i in range(L):
        snippet = _extract_snippet(
            recording=recording,
            t1=spike_train[i] - snippet_T1,
            t2=spike_train[i] + snippet_T2,
            recording_cache_dir=recording_cache_dir,
            chunk_size_sec=chunk_size_sec
        )
        ret[i, :, :] = snippet
    return ret

def _extract_snippet(*,
    recording: 'si.BaseRecording',
    t1: int,
    t2: int,
    recording_cache_dir: str,
    chunk_size_sec: float
):
    bytes_per_entry = 4
    num_channels = recording.get_num_channels()
    chunk_size_frames = int(chunk_size_sec * recording.get_sampling_frequency())
    chunk_ind1 = int(np.floor(t1 / chunk_size_frames))
    chunk_ind2 = int(np.floor(t2 / chunk_size_frames))
    if chunk_ind1 == chunk_ind2:
        chunk_fname = _make_chunk_file(recording=recording, chunk_ind=chunk_ind1, recording_cache_dir=recording_cache_dir, chunk_size_frames=chunk_size_frames)
        with open(chunk_fname, 'rb') as f:
            f.seek(bytes_per_entry * (t1 - chunk_ind1 * chunk_size_frames) * num_channels)
            data = f.read(bytes_per_entry * (t2 - t1) * num_channels)
            arr = np.frombuffer(data, dtype=np.float32)
            arr = arr.reshape((t2 - t1, num_channels))
            return arr
    else:
        raise Exception('Not implemented yet - and should not happen with subsampling strategy')

def _make_chunk_file(*,
    recording: 'si.BaseRecording',
    chunk_ind: int,
    recording_cache_dir: str,
    chunk_size_frames: int
):
    fname = f'{recording_cache_dir}/chunk_{chunk_ind}.bin'
    if os.path.exists(fname):
        return fname
    print(f'Creating chunk file: {fname}')
    traces = recording.get_traces(start_frame=chunk_ind * chunk_size_frames, end_frame=(chunk_ind + 1) * chunk_size_frames)
    traces = traces.astype(np.float32)
    traces.tofile(fname)
    assert os.path.exists(fname)
    return fname

def _get_neighborhood_channel_indices(*,
    recording: 'si.BaseRecording',
    peak_channel_index: int,
    max_num_channels_per_neighborhood: int
):
    num_channels = recording.get_num_channels()
    channel_locations = recording.get_channel_locations()
    dists = np.zeros((num_channels,))
    for i in range(num_channels):
        dists[i] = np.linalg.norm(channel_locations[i, :] - channel_locations[peak_channel_index, :])
    inds = [int(i) for i in np.argsort(dists)] # we don't want bigint
    ret = []
    for i in range(max_num_channels_per_neighborhood):
        if i < len(inds):
            ret.append(inds[i])
    return ret

def _format_unit_id(unit_id: int):
    # we dont want a bigint
    if isinstance(unit_id, str):
        return unit_id
    else:
        return int(unit_id)

def _compute_unit_location(*,
    average_waveform: np.ndarray,
    channel_locations: List[List[float]]
):
    abs_max_amplitudes = np.max(np.abs(average_waveform), axis=0)
    # center of mass
    d = len(channel_locations[0])
    weighted_sum_of_locations = np.zeros((d,))
    total_weight = 0
    for i in range(len(abs_max_amplitudes)):
        weighted_sum_of_locations += abs_max_amplitudes[i] * np.array(channel_locations[i])
        total_weight += abs_max_amplitudes[i]
    if total_weight == 0:
        return [0] * d
    else:
        return [float(a) for a in weighted_sum_of_locations / total_weight]

app.add_processor(CreateSpikeSortingAnalysisProcessor)

if __name__ == '__main__':
    app.run()
