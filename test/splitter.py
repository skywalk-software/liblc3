import wave
import numpy as np
import scipy.io.wavfile as wavfile
import os


def calculate_power(pcm, sr_hz):
    # Calculate the number of samples that correspond to 1kHz
    n_1khz = int(1000 * len(pcm) / sr_hz)

    # Perform FFT for all channels at once
    fft_vals = np.fft.rfft(pcm, axis=0)
    power_spectrum = np.abs(fft_vals) ** 2

    # Calculate power below and above 1kHz
    low_freq_power = np.sum(power_spectrum[:n_1khz], axis=0)
    high_freq_power = np.sum(power_spectrum[n_1khz:], axis=0)

    return low_freq_power, high_freq_power

def split_multichannel_wav(filename, output_dir, beamform=False, subtract_playback=True):
    # Open the multichannel wav file
    print("Opening " + filename)
    (sr_hz, masterpcm) = wavfile.read(filename)

    # Check if the file is multichannel
    if masterpcm.shape[1] > 1:
        print("Found " + str(masterpcm.shape[1]) + " channels")

        # Add new processing for channel 1 minus amplified channel 4
        if masterpcm.shape[1] >= 4 and subtract_playback:
            ch1 = masterpcm[:, 0]
            ch4 = masterpcm[:, 3]  # 0-based indexing, so channel 4 is index 3
            
            # Multiply ch4 by 10 and subtract from ch1
            processed = np.clip(ch1 - (ch4 * 10), -32768, 32767).astype(np.int16)
            
            # Write the processed channel
            base_name, ext = os.path.splitext(os.path.basename(filename))
            output_path = os.path.join(output_dir, f"{base_name}_ch1_minus_ch4{ext}")
            wavfile.write(output_path, sr_hz, processed)
            
            # Print stats for the new processed file
            print(f"Ch1 minus Ch4 pcm range: {np.min(processed)} to {np.max(processed)} pcm rms: {np.sqrt(np.mean(processed ** 2))}")
        
        # Calculate power in the signal above and below 1kHz
        low_freq_power, high_freq_power = calculate_power(masterpcm, sr_hz)

        # round it to a few decimal places
        print(f"Low frequency power: {low_freq_power}")
        print(f"High frequency power: {high_freq_power}")


        # Split each channel into a separate wav file
        for i in range(masterpcm.shape[1]):
            pcm = masterpcm[:, i]
            # increase volume by 20db
            pcm = np.clip(pcm * 12, -32768, 32767).astype(np.int16)

            # Write the single channel wav file
            base_name, ext = os.path.splitext(os.path.basename(filename))
            output_path = os.path.join(output_dir, f"{base_name}_ch{i}{ext}")
            # wavfile.write(output_path, sr_hz, pcm)

        if beamform:
            # Create delay-and-sum beamformed channels
            ch1 = masterpcm[:, 1]
            ch2 = masterpcm[:, 2]

            # Delay ch1 by one frame
            ch1_delayed = np.roll(ch1, 1)
            ch1_delayed[0] = 0  # Set the first element to 0 to avoid wrap-around

            # Delay ch2 by one frame
            ch2_delayed = np.roll(ch2, 1)
            ch2_delayed[0] = 0  # Set the first element to 0 to avoid wrap-around

            # Sum channels with delays
            combined_ch1_delayed = np.clip(ch1_delayed + ch2, -32768, 32767).astype(np.int16)
            combined_ch2_delayed = np.clip(ch1 + ch2_delayed, -32768, 32767).astype(np.int16)

            # Write the delay-and-sum beamformed channels
            output_path_ch1 = os.path.join(output_dir, f"{base_name}_ch1_delayed_sum{ext}")
            output_path_ch2 = os.path.join(output_dir, f"{base_name}_ch2_delayed_sum{ext}")
            wavfile.write(output_path_ch1, sr_hz, combined_ch1_delayed)
            wavfile.write(output_path_ch2, sr_hz, combined_ch2_delayed)
            # print stats for beamforming
            print(f"channel 1 delayed sum pcm range: {np.min(combined_ch1_delayed)} to {np.max(combined_ch1_delayed)} pcm rms: {np.sqrt(np.mean(combined_ch1_delayed ** 2))}")
            print(f"channel 2 delayed sum pcm range: {np.min(combined_ch2_delayed)} to {np.max(combined_ch2_delayed)} pcm rms: {np.sqrt(np.mean(combined_ch2_delayed ** 2))}")
    else:
        print("The file is not multichannel.")

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) != 2:
        print("Usage: python splitter.py <file_or_directory>")
        sys.exit(1)

    input_path = sys.argv[1]

    if os.path.isfile(input_path):
        # Handle single file input
        output_dir = os.path.join(os.path.dirname(input_path), "split")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        split_multichannel_wav(input_path, output_dir)
    elif os.path.isdir(input_path):
        # Handle directory input
        output_dir = os.path.join(input_path, "split")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for filename in os.listdir(input_path):
            if filename.endswith(".wav"):
                filepath = os.path.join(input_path, filename)
                split_multichannel_wav(filepath, output_dir)
    else:
        print("Invalid input. Please provide a valid file or directory.")
        sys.exit(1)