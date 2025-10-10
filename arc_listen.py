# Copyright 2025 Beau Ayres
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Proprietary Software - All Rights Reserved
#
# This software is proprietary and confidential. Unauthorized copying,
# distribution, modification, or use is strictly prohibited without
# express written permission from Beau Ayres.
#
# AGPL-3.0-or-later licensed
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3
# arc_listen.py - Real-time arc sound analysis for weld feedback.
# Integrates with arc_control for haptic guidance.

import numpy as np
import time
from ghost_hand import GhostHand
import pyaudio
import scipy.fft

class ArcListener:
    def __init__(self):
        self.hand = GhostHand(kappa=0.2)
        self.log = []
        self.CHUNK = 1024  # Audio samples per frame
        self.RATE = 16000  # Sample rate in Hz
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                channels=1,
                                rate=self.RATE,
                                input=True,
                                frames_per_buffer=self.CHUNK)
        self.running = True

    def analyze_sound(self, data):
        """Analyze frequency content of arc sound."""
        audio_data = np.frombuffer(data, dtype=np.float32)
        fft_data = np.abs(scipy.fft.fft(audio_data))
        freqs = np.fft.fftfreq(len(fft_data)) * self.RATE
        mask = freqs >= 0
        freqs = freqs[mask]
        fft_data = fft_data[mask]
        dominant_freq = freqs[np.argmax(fft_data[:len(freqs)//2])]  # Focus on lower half
        return dominant_freq

    def listen(self):
        """Listen to arc and trigger feedback."""
        while self.running:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                freq = self.analyze_sound(data)

                # Log frequency
                self.log.append((time.time(), freq))
                print(f"Dominant frequency: {freq:.1f} Hz")

                # Feedback based on frequency
                if freq < 2000:  # Too short arc
                    self.hand.pulse(1)
                    print("Arc too short - push in")
                elif freq > 8000:  # Spatter or too long
                    self.hand.pulse(2)
                    print("Arc too long/spatter - pull back")
                elif freq > 10000:  # Noise filter
                    print("Excess noise detected - ignoring")
                    continue

                time.sleep(0.01)  # 10ms loop
            except Exception as e:
                print(f"Audio error: {e}")
                break

    def stop(self):
        """Stop listening and clean up."""
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("Arc Listener Log:", self.log[-5:])  # Last 5 entries

if __name__ == "__main__":
    listener = ArcListener()
    try:
        listener.listen()
    except KeyboardInterrupt:
        listener.stop()
