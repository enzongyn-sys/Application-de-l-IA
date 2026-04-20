"""
Interface Gradio — Classificateur de Genres Musicaux GTZAN
Équipe n°1 : Jean-Maxime & Enzo
===========================================================
Usage dans Google Colab (après avoir exécuté le notebook complet) :

    !python gradio_app.py

Le script charge automatiquement les poids fine-tunés depuis ./distilhubert_gtzan/
"""

import os
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# ── Configuration ──────────────────────────────────────────────────
SR               = 16000
SEGMENT_DURATION = 5
N_SAMPLES_SEG    = SR * SEGMENT_DURATION
N_MELS           = 64
HOP_LENGTH       = 512
MODEL_NAME       = "ntu-spml/distilhubert"

GENRE_NAMES = ['blues', 'classical', 'country', 'disco',
               'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
N_CLASSES = len(GENRE_NAMES)

GENRE_EMOJI = {
    'blues': '🎸', 'classical': '🎻', 'country': '🤠', 'disco': '🕺',
    'hiphop': '🎤', 'jazz': '🎺', 'metal': '🤘', 'pop': '🎵',
    'reggae': '🌴', 'rock': '⚡'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

# ── Chargement du modèle ────────────────────────────────────────────
id2label = {i: g for i, g in enumerate(GENRE_NAMES)}
label2id = {g: i for i, g in enumerate(GENRE_NAMES)}

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=N_CLASSES,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Charger les poids fine-tunés si disponibles
checkpoint_dir = './distilhubert_gtzan'
loaded_finetuned = False
if os.path.exists(checkpoint_dir):
    checkpoints = sorted([
        d for d in os.listdir(checkpoint_dir)
        if d.startswith('checkpoint') and os.path.isdir(os.path.join(checkpoint_dir, d))
    ])
    if checkpoints:
        best_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])
        try:
            model = AutoModelForAudioClassification.from_pretrained(
                best_ckpt, ignore_mismatched_sizes=True)
            loaded_finetuned = True
            print(f"✅ Poids fine-tunés chargés : {best_ckpt}")
        except Exception as e:
            print(f"⚠️  Impossible de charger le checkpoint : {e}")

if not loaded_finetuned:
    print("⚠️  Aucun checkpoint trouvé — modèle pré-entraîné de base.")
    print("   → Exécuter d'abord l'étape 6 du notebook pour fine-tuner.")

model = model.eval().to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME, sampling_rate=SR)
print(f"✅ Modèle prêt ({sum(p.numel() for p in model.parameters()):,} paramètres)")


# ── Fonction de prédiction ──────────────────────────────────────────
def predict_genre(audio):
    if audio is None:
        return {}, None

    sr_input, audio_array = audio
    audio_float = audio_array.astype(np.float32)

    if audio_float.ndim == 2:
        audio_float = audio_float.mean(axis=1)

    if np.abs(audio_float).max() > 1.0:
        audio_float = audio_float / 32768.0

    if sr_input != SR:
        audio_float = librosa.resample(audio_float, orig_sr=sr_input, target_sr=SR)

    if len(audio_float) > N_SAMPLES_SEG:
        mid = len(audio_float) // 2
        start = max(0, mid - N_SAMPLES_SEG // 2)
        audio_float = audio_float[start:start + N_SAMPLES_SEG]
    if len(audio_float) < N_SAMPLES_SEG:
        audio_float = np.pad(audio_float, (0, N_SAMPLES_SEG - len(audio_float)))

    inputs = feature_extractor(
        audio_float, sampling_rate=SR,
        max_length=N_SAMPLES_SEG, truncation=True, padding=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        logits = model(inputs['input_values'].to(device)).logits
        probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    top_genre = GENRE_NAMES[probs.argmax()]
    scores = {
        f"{GENRE_EMOJI.get(GENRE_NAMES[i], '')} {GENRE_NAMES[i]}": float(probs[i])
        for i in range(N_CLASSES)
    }

    mel    = librosa.feature.melspectrogram(y=audio_float, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.5))
    img = librosa.display.specshow(mel_db, sr=SR, hop_length=HOP_LENGTH,
                                    x_axis='time', y_axis='mel', ax=ax1, cmap='magma')
    fig.colorbar(img, ax=ax1, format='%+2.0f dB')
    ax1.set_title(f'Mel-Spectrogramme — prédit : {top_genre.upper()} {GENRE_EMOJI.get(top_genre,"")}',
                  fontsize=12, fontweight='bold')

    colors = ['#e74c3c' if g == top_genre else '#3498db' for g in GENRE_NAMES]
    bars = ax2.barh([f"{GENRE_EMOJI.get(g,'')} {g}" for g in GENRE_NAMES],
                    probs, color=colors, edgecolor='white', linewidth=0.5)
    for bar, p in zip(bars, probs):
        if p > 0.02:
            ax2.text(p + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{p:.0%}', va='center', fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Probabilité')
    ax2.set_title('Distribution des genres', fontweight='bold')
    plt.tight_layout()

    return scores, fig


# ── Interface ───────────────────────────────────────────────────────
with gr.Blocks(title="🎵 Genre Musical Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎵 Classificateur de Genres Musicaux — GTZAN
    ### Un modèle de **parole** (DistilHuBERT) entraîné sur la **musique**
    > **Équipe n°1 — Jean-Maxime & Enzo** | Module 1 — Machine Listening — Track C

    Uploadez un fichier audio ou enregistrez avec votre microphone.
    Le modèle prédit le genre parmi : **Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock**

    💡 *Meilleurs résultats sur : Classical 🎻, Metal 🤘, Jazz 🎺, HipHop 🎤*
    """)

    with gr.Row():
        audio_input  = gr.Audio(sources=['upload', 'microphone'], type='numpy',
                                label='🎙️ Audio (upload ou micro)')
        genre_output = gr.Label(num_top_classes=5, label='🏆 Top 5 Genres prédits')

    spec_output = gr.Plot(label='📊 Mel-Spectrogramme & Distribution')
    predict_btn = gr.Button("🎯 Classifier le genre", variant='primary', size='lg')

    predict_btn.click(fn=predict_genre, inputs=audio_input, outputs=[genre_output, spec_output])
    audio_input.change(fn=predict_genre, inputs=audio_input, outputs=[genre_output, spec_output])

    gr.Markdown("""
    ---
    **Modèle** : [DistilHuBERT](https://huggingface.co/ntu-spml/distilhubert) fine-tuné sur
    [GTZAN](https://huggingface.co/datasets/marsyas/gtzan) — 999 clips · 10 genres · 5 epochs
    **Résultats** : RF=63.5% → CNN=71.2% → DistilHuBERT=70.4%
    """)


if __name__ == '__main__':
    demo.launch(share=True, server_name='0.0.0.0', server_port=7860)
