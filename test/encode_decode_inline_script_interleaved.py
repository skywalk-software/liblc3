import numpy as np
import scipy.io.wavfile as wavfile
import struct
import argparse
import os

## If having import issues with LC3, try running the following command in the terminal from test directory
# /path/to/python setup.py install
import lc3
import soundfile as sf

def encode_frames(dt, sr_hz, masterpcm, frame_nbytes):
    """Encode multi-channel audio frames using LC3"""
    if sr_hz not in (8000, 16000, 24000, 320000, 48000):
        raise ValueError('Unsupported input samplerate: %d' % sr_hz)
    if masterpcm.ndim != 2:
        print(masterpcm.shape)
        raise ValueError('Only multi channel wav file supported')

    ### Encoder Setup ###
    enc_c = lc3.setup_encoder(int(dt * 1000), sr_hz)
    frame_samples = int((dt * sr_hz) / 1000)
    num_channels = masterpcm.shape[1]

    # Store encoded data for all channels
    encoded_data_all_channels = []
    
    ### Encoding loop ###
    for ch in range(num_channels):
        # Set up byte array per channel
        pcm = masterpcm[:, ch]
        if len(pcm) % frame_samples > 0:
            pcm = np.append(pcm, np.zeros(frame_samples - (len(pcm) % frame_samples)))

        # Store encoded frames for this channel
        encoded_frames_this_channel = []
        
        # Run through each frame
        for i in range(0, len(pcm), frame_samples):
            # Encoding with C implementation
            frame_pcm = pcm[i:i+frame_samples]
            data_c = lc3.encode(enc_c, frame_pcm, frame_nbytes)
            
            # Store encoded frame data
            encoded_frames_this_channel.append(data_c)

        # Store all frames for this channel
        encoded_data_all_channels.append(encoded_frames_this_channel)

    print(f"\n✓ Encoding completed - {num_channels} channels, {len(encoded_data_all_channels[0])} frames")
    return encoded_data_all_channels


def interleave_frames(encoded_data_all_channels, frame_nbytes, output_file):
    """Interleave encoded frames across channels and save to file"""
    
    print('Interleaving and saving encoded frames...')
    with open(output_file, 'wb') as f:
        # Determine the number of frames (should be same for all channels)
        num_frames = len(encoded_data_all_channels[0])
        num_channels = len(encoded_data_all_channels)
        
        # Interleave bytes across channels for each frame
        for frame_idx in range(num_frames):
            # LC3 frames are always exactly frame_nbytes bytes
            # Interleave bytes: ch0byte, ch1byte, ch2byte, ch0byte, ...
            for byte_idx in range(frame_nbytes):
                for ch in range(num_channels):
                    f.write(encoded_data_all_channels[ch][frame_idx][byte_idx:byte_idx+1])

    print(f"✓ Interleaving completed - saved: {output_file}")
    return output_file


def deinterleave_frames(interleaved_file, frame_nbytes, num_channels):
    """De-interleave encoded frames from file"""
    
    # Read the interleaved encoded file
    print('Reading and de-interleaving encoded file...')
    with open(interleaved_file, 'rb') as f:
        interleaved_data = f.read()
    
    # Calculate frame info from file size
    total_bytes = len(interleaved_data)
    num_frames = total_bytes // (frame_nbytes * num_channels)
    
    print(f"De-interleaving: {num_frames} frames, {frame_nbytes} bytes/frame, {num_channels} channels")
    
    # De-interleave the data back to per-channel frames
    deinterleaved_data = [[] for _ in range(num_channels)]
    
    byte_offset = 0
    for frame_idx in range(num_frames):
        # Create frame data for each channel
        frame_data = [bytearray() for _ in range(num_channels)]
        
        # De-interleave bytes for this frame
        for byte_idx in range(frame_nbytes):
            for ch in range(num_channels):
                frame_data[ch].append(interleaved_data[byte_offset])
                byte_offset += 1
        
        # Convert to bytes and store
        for ch in range(num_channels):
            deinterleaved_data[ch].append(bytes(frame_data[ch]))

    print(f"✓ De-interleaving completed")
    return deinterleaved_data


def decode_frames(deinterleaved_data, dt, sr_hz):
    """Decode LC3 frames to PCM audio"""
    
    num_channels = len(deinterleaved_data)
    pcm_c = [np.array([], dtype=np.int16) for _ in range(num_channels)]
    
    # Decoder setup
    dec_c = lc3.setup_decoder(int(dt * 1000), sr_hz)
    
    for ch in range(num_channels):
        for frame_idx, frame_data in enumerate(deinterleaved_data[ch]):  
            # Decode with C implementation
            x_c = lc3.decode(dec_c, frame_data)
            pcm_c[ch] = np.concatenate((pcm_c[ch], x_c))

    print(f"\n✓ Decoding completed")
    return pcm_c


def encode_audio(dt, bitrate, name, sr_hz, masterpcm):
    """Encode multi-channel audio to interleaved LC3 format"""
    
    frame_nbytes = int((bitrate * dt) / (1000 * 8))
    
    # Step 1: Encode frames
    encoded_data_all_channels = encode_frames(dt, sr_hz, masterpcm, frame_nbytes)
    
    # Step 2: Interleave and save
    base_name, ext = os.path.splitext(name)
    interleaved_file = f"{base_name}_interleaved_channels.bin"
    interleave_frames(encoded_data_all_channels, frame_nbytes, interleaved_file)
    
    return interleaved_file


def decode_audio(interleaved_file, dt, bitrate, sr_hz, num_channels, output_name):
    """Decode interleaved LC3 format to multi-channel audio"""
    
    frame_nbytes = int((bitrate * dt) / (1000 * 8))
    
    # Step 1: De-interleave frames
    deinterleaved_data = deinterleave_frames(interleaved_file, frame_nbytes, num_channels)
    
    # Step 2: Decode frames
    pcm_c = decode_frames(deinterleaved_data, dt, sr_hz)

    # Step 3: Save combined multi-channel output
    pcm_c_array = np.stack(pcm_c, axis=-1).astype(np.int16)
    print(pcm_c_array.shape)
    print(f"pcm_c range: {np.min(pcm_c)} to {np.max(pcm_c)}")
    
    # Save final multi-channel output
    base_name, ext = os.path.splitext(output_name)
    ext = '.wav'
    output_file = f"{base_name}_lc3_{bitrate}{ext}"
    sf.write(output_file, pcm_c_array, sr_hz)
    
    print(f"✓ LC3 decode completed successfully")
    print(f"Files saved:")
    print(f"  - Combined output: {output_file}")
    
    return output_file


def encode_decode_audio(dt, bitrate, name, sr_hz, masterpcm):
    """Complete encode and decode workflow"""
    # Step 1: Encode to interleaved format
    interleaved_file = encode_audio(dt, bitrate, name, sr_hz, masterpcm)
    
    # Step 2: Decode from interleaved format (simulating client-side)
    output_file = decode_audio(interleaved_file, dt, bitrate, sr_hz, masterpcm.shape[1], name)
    
    return output_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LC3 C Implementation - Interleaved Output')
    parser.add_argument('wav_file',
        help='Input wave file', type=argparse.FileType('r'))
    parser.add_argument('--bitrate',
        help='Bitrate in bps', type=int, required=True)
    parser.add_argument('--dt',
        help='Frame duration in ms', type=float, default=10)
    args = parser.parse_args()

    # example parameters
    # args = argparse.Namespace(wav_file=open('test/16k.wav', 'r'), bitrate=32000, dt=10)

    if args.bitrate < 16000 or args.bitrate > 320000:
        raise ValueError('Invalid bitate %d bps' % args.bitrate)

    if args.dt not in (7.5, 10):
        raise ValueError('Invalid frame duration %.1f ms' % args.dt)

    (sr_hz, masterpcm) = wavfile.read(args.wav_file.name)

    encode_decode_audio(args.dt, args.bitrate, args.wav_file.name, sr_hz, masterpcm)
