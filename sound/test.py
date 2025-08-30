from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
from pydub import AudioSegment
import io
# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇧🇷 'p' => Brazilian Portuguese pt-br
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
pipeline = KPipeline(lang_code='a') # <= make sure lang_code matches voice, reference above.

# This text is for demonstration purposes only, unseen during training
text = '''
The rise of quantum computing threatens all current public-key schemes (RSA, ECC, etc.). Shor’s
algorithm can break these by efficiently solving factorization or discrete-logarithm problems, so
“harvest-now, decrypt-later” attacks could expose long-lived IoT data once quantum hardware
matures. Embedded devices (sensors, controllers, smart cards) often use weaker crypto already
and remain in service for many years, so upgrading to quantum-resistant algorithms is essential.
However, post-quantum (PQ) schemes typically have higher cost: larger key or signature sizes and
more computation . Constrained platforms (e.g. ARM Cortex‑M, RISC‑V micros) must balance security
against limited RAM/Flash, CPU speed, and energy. For example, a reference PQ library implementation
of CRYSTALS‑Kyber-512 on a 24 MHz Cortex‑M4 required ≈0.65 M CPU cycles for keygen and ≈0.96 M
for decapsulation, using ~9 KB RAM . Highly optimized assembly (“M4”) cuts this roughly in half and
reduces RAM to ~2.5 KB . In contrast, a hash-based signature like SPHINCS+ can take tens of
seconds to minutes to generate a signature on the same MCU , making it impractical for many IoT
uses. Thus, embedded PQC requires careful algorithm choice and implementation.
'''
# text = '「もしおれがただ偶然、そしてこうしようというつもりでなくここに立っているのなら、ちょっとばかり絶望するところだな」と、そんなことが彼の頭に思い浮かんだ。'
# text = '中國人民不信邪也不怕邪，不惹事也不怕事，任何外國不要指望我們會拿自己的核心利益做交易，不要指望我們會吞下損害我國主權、安全、發展利益的苦果！'
# text = 'Los partidos políticos tradicionales compiten con los populismos y los movimientos asamblearios.'
# text = 'Le dromadaire resplendissant déambulait tranquillement dans les méandres en mastiquant de petites feuilles vernissées.'
# text = 'ट्रांसपोर्टरों की हड़ताल लगातार पांचवें दिन जारी, दिसंबर से इलेक्ट्रॉनिक टोल कलेक्शनल सिस्टम'
# text = "Allora cominciava l'insonnia, o un dormiveglia peggiore dell'insonnia, che talvolta assumeva i caratteri dell'incubo."
# text = 'Elabora relatórios de acompanhamento cronológico para as diferentes unidades do Departamento que propõem contratos.'

# Clean the text by removing newlines and extra whitespace
text = ' '.join(text.split())

# Split text into sentences to handle length limits while maintaining coherence
import re
sentences = re.split(r'[.!?]+', text)
sentences = [s.strip() for s in sentences if s.strip()]

print(f"Processing {len(sentences)} sentences...")

# 4️⃣ Generate audio for each sentence and concatenate into single file
all_audio_segments = []
all_generated_text = []

for i, sentence in enumerate(sentences):
    print(f"Processing sentence {i+1}/{len(sentences)}: {sentence[:50]}...")

    generator = pipeline(
        sentence, voice='af_heart', # <= change voice here
        speed=1
    )

    # Generate audio for this sentence
    gs, ps, audio = next(generator)
    all_audio_segments.append(audio)
    all_generated_text.append(gs)
    print(f"Generated: {gs}")

# Concatenate all audio segments
import numpy as np
complete_audio = np.concatenate(all_audio_segments)

# Display and save the complete audio
print("Complete generated text:", ' '.join(all_generated_text))
display(Audio(data=complete_audio, rate=24000, autoplay=True))

# Save as WAV file
sf.write('output.wav', complete_audio, 24000)

# Convert to MP3 using pydub
# First, save the numpy array to a temporary WAV file in memory
temp_wav = io.BytesIO()
sf.write(temp_wav, complete_audio, 24000, format='WAV')
temp_wav.seek(0)

# Load with pydub and export as MP3
audio_segment = AudioSegment.from_wav(temp_wav)
audio_segment.export('output.mp3', format='mp3', bitrate='192k')

print("Audio saved as 'output.wav' and 'output.mp3'")
