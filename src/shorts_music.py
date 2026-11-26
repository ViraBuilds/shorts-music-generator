"""shorts-music — synthesise an upbeat chiptune loop for short videos.

Everything is generated from scratch with sine/triangle waves (NumPy) — no
copyrighted audio, no samples, no licence fees. Default output is a 20-second
mono 16-bit WAV: chords C - Am - F - G, a playful pentatonic melody, bass,
kick/hat drums, and optional chimes at given times (handy for scene cuts).

This is a simple loop generator, not a studio: one fixed chord pattern, one
fixed melody, one chiptune sound. If you want a different feel, edit
CHORD_ROOTS / CHORD_TONES / MELODY below.

Usage:
    python src/shorts_music.py                      # 20s -> music.wav
    python src/shorts_music.py -o intro.wav -d 10  # 10s -> intro.wav
    python src/shorts_music.py --no-chimes          # no scene-cut chimes
    python src/shorts_music.py --chimes 0,4,9       # chimes at 0s, 4s, 9s

Requirements: numpy (see requirements.txt).
"""

import argparse
import wave

import numpy as np

SR = 44100

# Chords (root MIDI notes): C=48, Am=45, F=41, G=43. One bar = 2.5 s.
CHORD_ROOTS = [48, 45, 41, 43]
CHORD_TONES = {48: [60, 64, 67], 45: [57, 60, 64], 41: [53, 57, 60], 43: [55, 59, 62]}

# Melody (MIDI notes), one per step across the track. 0 = rest.
# Playful pentatonic line, written by hand for the default 20 s track.
MELODY = [72, 76, 79, 76, 81, 79, 76, 72, 74, 77, 81, 77, 79, 76, 74, 72,
          72, 76, 79, 84, 79, 76, 74, 72, 77, 81, 79, 76, 74, 71, 72, 0]

BAR_LEN = 2.5          # seconds per chord bar
CHIME_NOTES = [84, 88, 91]  # bright arpeggio played at each chime time


def freq(midi):
    """MIDI note number -> frequency in Hz."""
    return 440.0 * 2 ** ((midi - 69) / 12)


class Track:
    """A mono float mix buffer with tiny synth helpers."""

    def __init__(self, duration):
        self.n = int(SR * duration)
        self.out = np.zeros(self.n)

    def note(self, midi, start, length, vol=0.25, wave_type="sine"):
        s = int(start * SR)
        e = min(self.n, int((start + length) * SR))
        if e <= s:
            return
        tt = np.arange(e - s) / SR
        f = freq(midi)
        if wave_type == "sine":
            sig = np.sin(2 * np.pi * f * tt)
        else:  # triangle-ish: triangle mixed with a little sine
            sig = 2 * np.abs(2 * ((f * tt) % 1) - 1) - 1
            sig = sig * 0.6 + 0.4 * np.sin(2 * np.pi * f * tt)
        env = np.minimum(1.0, tt / 0.01) * np.exp(-3.0 * tt / max(length, 0.05))
        self.out[s:e] += vol * sig * env

    def kick(self, start):
        s = int(start * SR)
        e = min(self.n, int((start + 0.18) * SR))
        if e <= s:
            return
        tt = np.arange(e - s) / SR
        f = 150 * np.exp(-tt * 22) + 45
        phase = 2 * np.pi * np.cumsum(f) / SR
        env = np.exp(-tt * 28)
        self.out[s:e] += 0.55 * np.sin(phase) * env

    def hat(self, start, vol=0.08):
        s = int(start * SR)
        e = min(self.n, int((start + 0.05) * SR))
        if e <= s:
            return
        rng = np.random.default_rng(int(start * 1000) + 7)
        self.out[s:e] += vol * rng.standard_normal(e - s) * np.exp(-np.arange(e - s) / SR * 90)

    def render(self, duration, chimes):
        """Fill the buffer: melody, chord pads, bass, drums, chimes."""
        # Melody: one note per step across the whole track.
        step = duration / len(MELODY)
        for i, m in enumerate(MELODY):
            if m:
                self.note(m, i * step, step * 0.95, vol=0.20, wave_type="tri")

        # Bars: chord pad + bass + kick/hat on each beat (0.625 s grid).
        n_bars = max(1, round(duration / BAR_LEN))
        for bar in range(n_bars):
            start = bar * BAR_LEN
            if start >= duration:
                break
            root = CHORD_ROOTS[bar % len(CHORD_ROOTS)]
            for ct in CHORD_TONES[root]:
                self.note(ct, start, 2.4, vol=0.07)
            for b in range(4):
                beat = start + b * 0.625
                if beat >= duration:
                    break
                self.note(root - 12, beat, 0.55, vol=0.28)
                self.kick(beat)
                self.hat(beat + 0.3125)

        # Scene-cut chimes.
        for cut in chimes:
            if cut >= duration:
                continue
            for i, m in enumerate(CHIME_NOTES):
                self.note(m, cut + i * 0.07, 0.5, vol=0.16)

        # Gentle fade-in at the start, fade-out over the last second.
        self.out[:int(0.05 * SR)] *= np.linspace(0, 1, int(0.05 * SR))
        self.out[-int(1.0 * SR):] *= np.linspace(1, 0, int(1.0 * SR))

    def to_wav_bytes(self):
        peak = np.max(np.abs(self.out))
        norm = (self.out / peak * 0.85).astype(np.float32) if peak > 0 else self.out
        return (norm * 32767).astype(np.int16).tobytes(), peak


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Synthesise an upbeat chiptune loop (WAV) for short videos.")
    p.add_argument("-d", "--duration", type=float, default=20.0,
                   help="track length in seconds (default: 20)")
    p.add_argument("-o", "--output", default="music.wav",
                   help="output WAV file (default: music.wav)")
    p.add_argument("--chimes", default="0,5,12",
                   help="comma-separated seconds for scene-cut chimes "
                        "(default: 0,5,12)")
    p.add_argument("--no-chimes", action="store_true",
                   help="skip the scene-cut chimes")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.duration <= 0:
        raise SystemExit("duration must be positive")
    chimes = [] if args.no_chimes else [float(c) for c in args.chimes.split(",") if c.strip()]

    track = Track(args.duration)
    track.render(args.duration, chimes)
    pcm, peak = track.to_wav_bytes()
    with wave.open(args.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    print(f"wrote {args.output} ({args.duration:g}s, peak {peak:.3f})")


if __name__ == "__main__":
    main()
