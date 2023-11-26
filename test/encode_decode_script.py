import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import struct
import argparse
import os
from multiprocessing import Process
import lc3
import tables as T, appendix_c as C

import encoder
import decoder



def encode_audio(dt, bitrate, name, sr_hz, pcm, ch_index):
    if sr_hz not in (8000, 16000, 24000, 320000, 48000):
        raise ValueError('Unsupported input samplerate: %d' % sr_hz)
    if pcm.ndim != 1:
        raise ValueError('Only single channel wav file supported')

    ### Setup ###

    enc = encoder.Encoder(dt, sr_hz)
    enc_c = lc3.setup_encoder(int(dt * 1000), sr_hz)

    frame_samples = int((dt * sr_hz) / 1000)
    frame_nbytes = int((bitrate * dt) / (1000 * 8))

    ### File Header ###
    base_name, ext = os.path.splitext(name)
    f_py = open(f"{base_name}_ch{ch_index}{ext}", 'wb') if name else None

    header = struct.pack('=HHHHHHHI', 0xcc1c, 18,
        sr_hz // 100, bitrate // 100, 1, int(dt * 100), 0, len(pcm))

    if f_py: f_py.write(header)

    ### Encoding loop ###

    if len(pcm) % frame_samples > 0:
        pcm = np.append(pcm, np.zeros(frame_samples - (len(pcm) % frame_samples)))
        
    for i in range(0, len(pcm), frame_samples):

        print('Encoding frame %d' % (i // frame_samples), end='\r')

        frame_pcm = pcm[i:i+frame_samples]

        data = enc.run(frame_pcm, frame_nbytes)
        data_c = lc3.encode(enc_c, frame_pcm, frame_nbytes)

        if f_py: 
            f_py.write(struct.pack('=H', frame_nbytes))
            f_py.write(data)

    print('done ! %16s' % '')

    ### Terminate ###
    if f_py: f_py.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LC3 Encoder Test Framework')
    parser.add_argument('wav_file',
        help='Input wave file', type=argparse.FileType('r'))
    parser.add_argument('--bitrate',
        help='Bitrate in bps', type=int, required=True)
    parser.add_argument('--dt',
        help='Frame duration in ms', type=float, default=10)
    parser.add_argument('--pyout',
        help='Python output file', type=argparse.FileType('w'))
    args = parser.parse_args()

    if args.bitrate < 16000 or args.bitrate > 320000:
        raise ValueError('Invalid bitate %d bps' % args.bitrate)

    if args.dt not in (7.5, 10):
        raise ValueError('Invalid frame duration %.1f ms' % args.dt)

    (sr_hz, masterpcm) = wavfile.read(args.wav_file.name)

    processes = []
    for i in range(masterpcm.shape[1]):
        pcm = masterpcm[:, i]
        p = Process(target=encode_audio, args=(args.dt, args.bitrate, args.pyout.name, sr_hz, pcm, i))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    else:
        pass


    # Decoding
    # Currently using the saved encoded output
    for i in range(masterpcm.shape[1]):
        base_name, ext = os.path.splitext(args.pyout.name)
        print(f"{base_name}_ch{i}{ext}")
        f_lc3 = open(f"{base_name}_ch{i}{ext}", 'rb') if args.pyout else None

        header = struct.unpack('=HHHHHHHI', f_lc3.read(18))

        if header[0] != 0xcc1c:
            raise ValueError('Invalid bitstream file')

        if header[4] != 1:
            raise ValueError('Unsupported number of channels')

        sr_hz = header[2] * 100
        bitrate = header[3] * 100
        nchannels = header[4]
        dt_ms = header[5] / 100

        f_lc3.seek(header[1])

        ### Setup ###

        dec = decoder.Decoder(dt_ms, sr_hz)
        dec_c = lc3.setup_decoder(int(dt_ms * 1000), sr_hz)

        pcm_c  = np.empty(0).astype(np.int16)
        pcm_py = np.empty(0).astype(np.int16)

        ### Decoding loop ###

        nframes = 0

        while True:

            data = f_lc3.read(2)
            if len(data) != 2:
                break

            (frame_nbytes,) = struct.unpack('=H', data)

            print('Decoding frame %d' % nframes, end='\r')

            data = f_lc3.read(frame_nbytes)

            x = dec.run(data)
            pcm_py = np.append(pcm_py,
                np.clip(np.round(x), -32768, 32767).astype(np.int16))

            x_c = lc3.decode(dec_c, data)
            pcm_c = np.append(pcm_c, x_c)

            nframes += 1

        print('done ! %16s' % '')

    ### Terminate ###
    # print range of values in pcm_py
    print(f"pcm_py range: {np.min(pcm_py)} to {np.max(pcm_py)}")
    if args.pyout:
        base_name, ext = os.path.splitext(args.pyout.name)
        ext = '.wav'
        wavfile.write(f"{base_name}_out{ext}", sr_hz, pcm_py)

### ------------------------------------------------------------------------ ###



### ------------------------------------------------------------------------ ###
