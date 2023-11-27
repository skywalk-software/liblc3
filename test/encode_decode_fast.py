import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import struct
import argparse
import os
from multiprocessing import Process, Pool
import lc3
import tables as T, appendix_c as C

import encoder
import decoder


def process_channel(args):
    pcm, frame_samples, frame_nbytes, enc, enc_c, dec, dec_c = args

    if len(pcm) % frame_samples > 0:
        pcm = np.append(pcm, np.zeros(frame_samples - (len(pcm) % frame_samples)))

    pcm_py = np.array([], dtype=np.int16)
    pcm_c = np.array([], dtype=np.int16)

    for i in range(0, len(pcm), frame_samples):
        # Encoding
        frame_pcm = pcm[i:i+frame_samples]
        data = enc.run(frame_pcm, frame_nbytes)
        data_c = lc3.encode(enc_c, frame_pcm, frame_nbytes)

        # Decoding
        x = dec.run(data)
        pcm_py = np.concatenate((pcm_py, np.clip(np.round(x), -32768, 32767).astype(np.int16)))
        x_c = lc3.decode(dec_c, data_c)
        pcm_c = np.concatenate((pcm_c, x_c))

    return pcm_py, pcm_c


def encode_decode_audio(dt, bitrate, name, sr_hz, masterpcm):
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

    with Pool() as pool:
        args = [(masterpcm[:, ch], frame_samples, frame_nbytes, enc, enc_c, dec, dec_c) for ch in range(masterpcm.shape[1])]
        results = pool.map(process_channel, args)

    pcm_py = [result[0] for result in results]
    pcm_c = [result[1] for result in results]

    pcm_py_array = np.stack(pcm_py, axis=-1).astype(np.int16)
    pcm_c_array = np.stack(pcm_c, axis=-1).astype(np.int16)
    print('done ! %16s' % '')

    base_name, ext = os.path.splitext(name)
    ext = '.wav'
    wavfile.write(f"{base_name}_combined{ext}", sr_hz, pcm_py_array)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LC3 Encoder Test Framework')
    parser.add_argument('wav_file',
        help='Input wave file', type=argparse.FileType('r'))
    parser.add_argument('--bitrate',
        help='Bitrate in bps', type=int, required=True)
    parser.add_argument('--dt',
        help='Frame duration in ms', type=float, default=10)
    args = parser.parse_args()

    if args.bitrate < 16000 or args.bitrate > 320000:
        raise ValueError('Invalid bitate %d bps' % args.bitrate)

    if args.dt not in (7.5, 10):
        raise ValueError('Invalid frame duration %.1f ms' % args.dt)

    (sr_hz, masterpcm) = wavfile.read(args.wav_file.name)
    encode_decode_audio(args.dt, args.bitrate, args.wav_file.name, sr_hz, masterpcm)
        
### ------------------------------------------------------------------------ ###
