import wave
import numpy as np
import scipy.io.wavfile as wavfile
import os
import sys


def combine_multichannel_wav(wav1, wav2, wav3):
    # Read the wav files
    (sr_hz1, pcm1) = wavfile.read(wav1)
    (sr_hz2, pcm2) = wavfile.read(wav2)
    (sr_hz3, pcm3) = wavfile.read(wav3)

    # Check if the sample rates are the same
    if sr_hz1 != sr_hz2 or sr_hz1 != sr_hz3:
        print("Sample rates are not the same.")
        return

    # Combine the waveforms
    combined_pcm = np.stack((pcm1, pcm2, pcm3), axis=-1)

    # Write the combined wav file
    base_name, ext = os.path.splitext(wav1)
    wavfile.write(f"{base_name}_combined{ext}", sr_hz1, combined_pcm)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python Untitled-1.py <wav1> <wav2> <wav3>")
        sys.exit(1)
    combine_multichannel_wav(sys.argv[1], sys.argv[2], sys.argv[3])
