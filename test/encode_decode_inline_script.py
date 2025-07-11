import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import struct
import argparse
import os
from multiprocessing import Process

## If having import issues with LC3, try running the following command in the terminal from test directory
# /path/to/python setup.py install
import lc3
import tables as T, appendix_c as C
import soundfile as sf

import encoder
import decoder



def encode_decode_audio(dt, bitrate, name, sr_hz, masterpcm, save_encoded=False):
    if sr_hz not in (8000, 16000, 24000, 320000, 48000):
        raise ValueError('Unsupported input samplerate: %d' % sr_hz)
    if masterpcm.ndim != 2:
        print(masterpcm.shape)
        raise ValueError('Only multi channel wav file supported')

    ### Encoder Setup ###
    enc = encoder.Encoder(dt, sr_hz)
    enc_c = lc3.setup_encoder(int(dt * 1000), sr_hz)

    ### Decoder Setup ###
    dec = decoder.Decoder(dt, sr_hz)
    dec_c = lc3.setup_decoder(int(dt * 1000), sr_hz)

    frame_samples = int((dt * sr_hz) / 1000)
    frame_nbytes = int((bitrate * dt) / (1000 * 8))

    # pcm_c  = np.empty((3,0)).astype(np.int16)
    # pcm_py = np.empty((3,0)).astype(np.int16)
    pcm_py = [np.array([], dtype=np.int16) for _ in range(3)]
    pcm_c = [np.array([], dtype=np.int16) for _ in range(3)]

    ### Encoding + Decoding loop ###
    for ch in range(masterpcm.shape[1]):
        # Set up byte array per channel
        pcm = masterpcm[:, ch]
        if len(pcm) % frame_samples > 0:
            pcm = np.append(pcm, np.zeros(frame_samples - (len(pcm) % frame_samples)))

        ### File Header ###
        if save_encoded:
            base_name, ext = os.path.splitext(name)
            ext = '.bin'
            f_py = open(f"{base_name}_lc3_{bitrate}_ch{ch}{ext}", 'wb')
            header = struct.pack('=HHHHHHHI', 0xcc1c, 18,
            sr_hz // 100, bitrate // 100, 1, int(dt * 100), 0, len(pcm))
            f_py.write(header)
        
        # Run through each frame
        for i in range(0, len(pcm), frame_samples):
            # Encoding
            print('Encoding frame %d' % (i // frame_samples), end='\r')
            frame_pcm = pcm[i:i+frame_samples]
            data = enc.run(frame_pcm, frame_nbytes)
            data_c = lc3.encode(enc_c, frame_pcm, frame_nbytes)
            if save_encoded: 
                f_py.write(struct.pack('=H', frame_nbytes))
                f_py.write(data)

            # Decoding
            print('Decoding frame %d' % (i // frame_samples), end='\r')
            x = dec.run(data)
            # pcm_py[ch] = np.append(pcm_py[ch],
            #     np.clip(np.round(x), -32768, 32767).astype(np.int16))
            pcm_py[ch] = np.concatenate((pcm_py[ch], np.clip(np.round(x), -32768, 32767).astype(np.int16)))
            x_c = lc3.decode(dec_c, data_c)
            # pcm_c[ch] = np.append(pcm_c[ch], x_c)
            pcm_c[ch] = np.concatenate((pcm_c[ch], x_c))

        # Save each channel as a separate .wav file
        channel_wav_name = f"{base_name}_lc3_{bitrate}_ch{ch}.wav"
        sf.write(channel_wav_name, pcm_py[ch], sr_hz)

    pcm_py_array = np.stack(pcm_py, axis=-1).astype(np.int16)
    pcm_c_array = np.stack(pcm_c, axis=-1).astype(np.int16)
    print('done ! %16s' % '')
    print(pcm_py_array.shape)
    print(f"pcm_py range: {np.min(pcm_py)} to {np.max(pcm_py)}")
    base_name, ext = os.path.splitext(name)
    ext = '.wav'
    sf.write(f"{base_name}_lc3_{bitrate}{ext}", pcm_py_array, sr_hz)

    ### Terminate ###
    if save_encoded: f_py.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LC3 Encoder Test Framework')
    parser.add_argument('wav_file',
        help='Input wave file', type=argparse.FileType('r'))
    parser.add_argument('--bitrate',
        help='Bitrate in bps', type=int, required=True)
    parser.add_argument('--dt',
        help='Frame duration in ms', type=float, default=10)
    parser.add_argument('--save_encoded', action='store_true')
    args = parser.parse_args()

    # example parameters
    # args = argparse.Namespace(wav_file=open('test/16k.wav', 'r'), bitrate=32000, dt=10, save_encoded=False)


    if args.bitrate < 16000 or args.bitrate > 320000:
        raise ValueError('Invalid bitate %d bps' % args.bitrate)

    if args.dt not in (7.5, 10):
        raise ValueError('Invalid frame duration %.1f ms' % args.dt)

    (sr_hz, masterpcm) = wavfile.read(args.wav_file.name)

    processes = []
    encode_decode_audio(args.dt, args.bitrate, args.wav_file.name, sr_hz, masterpcm, save_encoded=args.save_encoded)
        
    #     p = Process(target=encode_audio, args=(args.dt, args.bitrate, args.pyout.name, sr_hz, pcm, i))
    #     p.start()
    #     processes.append(p)

    # # Wait for all processes to finish
    # for p in processes:
    #     p.join()

### ------------------------------------------------------------------------ ###