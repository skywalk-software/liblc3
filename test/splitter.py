import wave
import numpy as np
import scipy.io.wavfile as wavfile
import os

def split_multichannel_wav(filename):
    # Open the multichannel wav file
    (sr_hz, masterpcm) = wavfile.read(filename)
    
    # Check if the file is multichannel
    if masterpcm.shape[1] > 1:
        print("Found " + str(masterpcm.shape[1]) + " channels")
        
        # Split each channel into a separate wav file
        for i in range(masterpcm.shape[1]):
            pcm = masterpcm[:, i]
            # print stats for each pcm channel
            print(f"channel {i} pcm range: {np.min(pcm)} to {np.max(pcm)} pcm rms: {np.sqrt(np.mean(pcm**2))}")
            
            # Write the single channel wav file
            base_name, ext = os.path.splitext(filename)
            wavfile.write(f"{base_name}_ch{i}{ext}", sr_hz, pcm)
    else:
        print("The file is not multichannel.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python splitter.py <filename>")
        sys.exit(1)
    split_multichannel_wav(sys.argv[1])

