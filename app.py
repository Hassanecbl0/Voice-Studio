import os
import socket
import sys
import tempfile

# Configure Windows DLL directories for PyTorch before any other imports
if sys.platform == "win32":
    project_root = os.path.dirname(os.path.abspath(__file__))
    torch_lib = os.path.join(project_root, "venv", "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(torch_lib):
        try:
            _torch_dll_handle = os.add_dll_directory(torch_lib)
        except Exception:
            pass
        if torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# CRITICAL: Import torch and torchaudio before any other packages (including Gradio)
# to avoid OpenMP / C-runtime DLL symbol initialization collisions on Windows
import torch
import torchaudio
import soundfile as sf

# In PyTorch 2.6+, torch.load defaults weights_only=True, which breaks Coqui TTS checkpoint loading.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

# Fix torchaudio 2.11+ torchcodec requirement
def _soundfile_load(filepath, **kwargs):
    data, sr = sf.read(filepath)
    if data.ndim == 1:
        tensor = torch.from_numpy(data).unsqueeze(0).float()
    else:
        tensor = torch.from_numpy(data.T).float()
    return tensor, sr

def _soundfile_save(filepath, src, sample_rate, **kwargs):
    if isinstance(src, torch.Tensor):
        src_np = src.detach().cpu().numpy()
        if src_np.ndim == 2:
            src_np = src_np.T
    else:
        src_np = src
    sf.write(filepath, src_np, sample_rate)

torchaudio.load = _soundfile_load
torchaudio.save = _soundfile_save

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from core.audio_utils import (
    preprocess_reference_audio,
    get_preset_voices,
    save_new_preset,
    ensure_dir,
)
from core.tts_engine import XTTSEngine, SUPPORTED_LANGUAGES
from core.transcriber import WhisperTranscriber

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

ensure_dir(VOICES_DIR)
ensure_dir(OUTPUTS_DIR)

# Initialize engines lazily
tts_engine = XTTSEngine()
transcriber = WhisperTranscriber(model_size="base")

def get_voice_choices():
    """Fetch available voices for dropdowns."""
    presets = get_preset_voices(VOICES_DIR)
    return list(presets.keys())

# ---------------- CORE HANDLERS (NO EMOJIS) ----------------

def handle_tts(
    voice_source_type: str,
    selected_preset: str,
    uploaded_reference: str,
    text_input: str,
    language_label: str,
    speed: float,
    temperature: float,
    repetition_penalty: float,
):
    """Generate cloned speech from text."""
    if not text_input or not text_input.strip():
        raise gr.Error("Veuillez saisir le texte que la voix doit énoncer.")

    ref_audio_path = None
    if voice_source_type == "Voix enregistrée":
        presets = get_preset_voices(VOICES_DIR)
        if not selected_preset or selected_preset not in presets:
            raise gr.Error("Veuillez sélectionner une voix de référence.")
        ref_audio_path = presets[selected_preset]
    else:
        if not uploaded_reference:
            raise gr.Error("Veuillez importer ou enregistrer un extrait vocal de référence.")
        ref_audio_path = uploaded_reference

    cleaned_ref = tempfile.mktemp(suffix=".wav")
    preprocess_reference_audio(ref_audio_path, cleaned_ref)
    lang_code = SUPPORTED_LANGUAGES.get(language_label, "en")

    try:
        gr.Info("Synthèse vocale en cours...")
        result = tts_engine.clone_voice(
            text=text_input.strip(),
            speaker_wav=cleaned_ref,
            language=lang_code,
            speed=speed,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        status_msg = f"Synthèse terminée en {result['elapsed_sec']}s ({result['character_count']} caractères)"
        return result["output_path"], status_msg
    except Exception as e:
        raise gr.Error(f"Erreur de synthèse : {str(e)}")
    finally:
        if os.path.exists(cleaned_ref):
            try:
                os.remove(cleaned_ref)
            except Exception:
                pass

def handle_transcription(input_speech_audio: str, source_language_label: str):
    """Step 1 of Speech-to-Speech: Transcribe source speech with Whisper."""
    if not input_speech_audio:
        raise gr.Error("Veuillez enregistrer ou importer un fichier audio.")

    gr.Info("Transcription audio en cours...")
    try:
        source_language = SUPPORTED_LANGUAGES.get(source_language_label)
        res = transcriber.transcribe(input_speech_audio, language=source_language)
        conf = int(res['language_probability'] * 100)
        status_text = f"Langue détectée : {res['language'].upper()} ({conf}% confiance)"
        return res["text"], status_text
    except Exception as e:
        raise gr.Error(f"Erreur de transcription : {str(e)}")

def handle_speech_to_speech(
    source_audio: str,
    transcribed_text: str,
    target_voice_type: str,
    selected_preset: str,
    uploaded_target_ref: str,
    language_label: str,
    speed: float,
    temperature: float,
    repetition_penalty: float,
):
    """Step 2 of Speech-to-Speech: Synthesize in target cloned voice."""
    if not transcribed_text or not transcribed_text.strip():
        raise gr.Error("Aucun texte à transformer.")

    target_ref_path = None
    if target_voice_type == "Voix enregistrée":
        presets = get_preset_voices(VOICES_DIR)
        if not selected_preset or selected_preset not in presets:
            raise gr.Error("Veuillez sélectionner une voix cible.")
        target_ref_path = presets[selected_preset]
    else:
        if not uploaded_target_ref:
            raise gr.Error("Veuillez importer ou enregistrer un extrait vocal cible.")
        target_ref_path = uploaded_target_ref

    cleaned_ref = tempfile.mktemp(suffix=".wav")
    preprocess_reference_audio(target_ref_path, cleaned_ref)
    lang_code = SUPPORTED_LANGUAGES.get(language_label, "en")

    try:
        gr.Info("Transformation vocale en cours...")
        result = tts_engine.clone_voice(
            text=transcribed_text.strip(),
            speaker_wav=cleaned_ref,
            language=lang_code,
            speed=speed,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        status_msg = f"Transformation terminée en {result['elapsed_sec']}s"
        return source_audio, result["output_path"], status_msg
    except Exception as e:
        raise gr.Error(f"Erreur de transformation : {str(e)}")
    finally:
        if os.path.exists(cleaned_ref):
            try:
                os.remove(cleaned_ref)
            except Exception:
                pass

def handle_save_voice(uploaded_audio: str, voice_name: str):
    """Save a new voice to preset library."""
    if not uploaded_audio:
        raise gr.Error("Veuillez importer un fichier audio.")
    if not voice_name or not voice_name.strip():
        raise gr.Error("Veuillez donner un nom à la voix.")

    save_new_preset(uploaded_audio, voice_name.strip(), VOICES_DIR)
    choices = get_voice_choices()
    return (
        f"Voix '{voice_name.strip()}' enregistrée avec succès.",
        gr.update(choices=choices, value=choices[0] if choices else None),
        gr.update(choices=choices, value=choices[0] if choices else None),
    )

# ---------------- FLUID, CENTERED & EMOJI-FREE CSS ----------------

FLUID_CENTERED_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --bg-app: #090D14;
  --bg-card: #111726;
  --bg-card-subtle: #141C2E;
  --bg-input: #0A0E17;
  --border-subtle: rgba(255, 255, 255, 0.08);
  --border-focus: #6366F1;
  --accent-primary: #6366F1;
  --accent-gradient: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
  --accent-glow: 0 4px 20px -2px rgba(99, 102, 241, 0.4);
  --text-main: #F8FAFC;
  --text-secondary: #94A3B8;
  --text-muted: #64748B;
  --radius-card: 16px;
  --radius-btn: 12px;
  --radius-input: 10px;
}

body, html {
  background-color: var(--bg-app) !important;
  color: var(--text-main) !important;
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  margin: 0 !important;
  padding: 0 !important;
  min-height: 100vh !important;
}

.gradio-container {
  background-color: var(--bg-app) !important;
  max-width: 100% !important;
  width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: flex-start !important;
}

footer {
  display: none !important;
}

/* Centered Main Application Shell (max-w-7xl) */
.app-wrapper {
  width: 100% !important;
  max-width: 1240px !important;
  margin: 0 auto !important;
  padding: 24px 20px 48px 20px !important;
  box-sizing: border-box !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 20px !important;
}

/* Master Top Header */
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-card);
  box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 1px 1px rgba(255, 255, 255, 0.05);
}

.brand-section {
  display: flex;
  align-items: center;
  gap: 12px;
}
.brand-logo-icon {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  background: var(--accent-gradient);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: var(--accent-glow);
  color: #FFFFFF;
}
.brand-text-wrap {
  display: flex;
  flex-direction: column;
}
.brand-title {
  font-size: 16px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: #FFFFFF;
}
.brand-subtitle {
  font-size: 11px;
  color: var(--text-secondary);
  font-weight: 500;
}

.header-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  background: rgba(16, 185, 129, 0.08);
  border: 1px solid rgba(16, 185, 129, 0.2);
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  color: #34D399;
}
.status-dot-pulse {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #10B981;
}

/* Studio 2-Panel Content Grid */
.studio-layout-grid {
  display: grid !important;
  grid-template-columns: 380px 1fr !important;
  gap: 20px !important;
  align-items: stretch !important;
  width: 100% !important;
}

/* Fluid Cards */
.studio-card {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-card) !important;
  box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.4), 0 0 1px 1px rgba(255, 255, 255, 0.04) !important;
  padding: 22px 24px !important;
  box-sizing: border-box !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 16px !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.card-heading {
  font-size: 13px;
  font-weight: 700;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 2px;
}

/* Gradio Component Clean Overrides */
.studio-card .block, .studio-card .form {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  margin: 0 !important;
}

/* Modern Segmented Tabs */
div[role="tablist"] {
  display: flex !important;
  background: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: 14px !important;
  padding: 5px !important;
  gap: 6px !important;
  margin-bottom: 0 !important;
}
div[role="tablist"] button[role="tab"] {
  background: transparent !important;
  border: none !important;
  border-radius: 10px !important;
  color: var(--text-secondary) !important;
  font-size: 13.5px !important;
  font-weight: 500 !important;
  padding: 8px 18px !important;
  transition: all 0.2s ease !important;
}
div[role="tablist"] button[role="tab"]:hover {
  color: #FFFFFF !important;
}
div[role="tablist"] button[role="tab"].selected {
  background: var(--bg-card-subtle) !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}

/* Textareas & Inputs */
textarea, input[type="text"] {
  background-color: var(--bg-input) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-input) !important;
  color: var(--text-main) !important;
  font-family: inherit !important;
  font-size: 14px !important;
  line-height: 1.6 !important;
  padding: 12px 14px !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
textarea:focus, input[type="text"]:focus {
  border-color: var(--border-focus) !important;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}

/* Primary Generation CTA */
button.primary-btn, button.primary {
  background: var(--accent-gradient) !important;
  border: none !important;
  border-radius: var(--radius-btn) !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
  font-size: 14.5px !important;
  padding: 13px 24px !important;
  box-shadow: var(--accent-glow) !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
  cursor: pointer !important;
  width: 100% !important;
}
button.primary-btn:hover, button.primary:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 24px -2px rgba(99, 102, 241, 0.55) !important;
}

/* Secondary Button */
button.secondary-btn, button.secondary {
  background: var(--bg-input) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-btn) !important;
  color: var(--text-main) !important;
  font-weight: 600 !important;
  font-size: 13.5px !important;
  padding: 10px 18px !important;
  transition: all 0.2s ease !important;
}
button.secondary-btn:hover, button.secondary:hover {
  background: #1A2234 !important;
  border-color: rgba(255, 255, 255, 0.15) !important;
}

/* Discreet Status Badge */
.status-pill-result {
  font-size: 12.5px;
  color: #A5B4FC;
  padding: 4px 0;
  margin: 0;
  font-weight: 500;
}

/* Sliders */
input[type="range"] {
  accent-color: var(--accent-primary) !important;
}

/* Responsive Media Queries */
@media (max-width: 960px) {
  .app-wrapper {
    padding: 16px 12px 32px 12px !important;
  }
  .studio-layout-grid {
    grid-template-columns: 1fr !important;
    gap: 16px !important;
  }
}
"""

def build_ui():
    initial_voices = get_voice_choices()
    languages = list(SUPPORTED_LANGUAGES.keys())

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "Inter", "sans-serif"],
    ).set(
        body_background_fill="#090D14",
        body_background_fill_dark="#090D14",
        block_background_fill="#111726",
        block_background_fill_dark="#111726",
        block_border_color="rgba(255, 255, 255, 0.08)",
        block_border_color_dark="rgba(255, 255, 255, 0.08)",
        block_radius="16px",
        input_background_fill="#0A0E17",
        input_background_fill_dark="#0A0E17",
        input_border_color="rgba(255, 255, 255, 0.08)",
        input_border_color_dark="rgba(255, 255, 255, 0.08)",
    )

    with gr.Blocks(title="VoiceFlow Studio") as demo:
        gr.HTML(f"<style>{FLUID_CENTERED_CSS}</style>")

        # Centered Container Wrapper
        with gr.Column(elem_classes=["app-wrapper"]):
            
            # Header
            gr.HTML(
                """
                <header class="app-header">
                    <div class="brand-section">
                        <div class="brand-logo-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M12 2v20M17 5v14M7 5v14M22 10v4M2 10v4"/>
                            </svg>
                        </div>
                        <div class="brand-text-wrap">
                            <span class="brand-title">VoiceFlow Studio</span>
                            <span class="brand-subtitle">Synthèse Neuronale XTTS-v2</span>
                        </div>
                    </div>
                    <div class="header-status">
                        <span class="status-dot-pulse"></span>
                        <span>Moteur Actif</span>
                    </div>
                </header>
                """
            )

            # Segmented Workspace Tabs
            with gr.Tabs():

                # ================= TAB 1: SYNTHÈSE VOCALE =================
                with gr.TabItem("Synthèse Vocale"):
                    with gr.Row(elem_classes=["studio-layout-grid"]):
                        
                        # Colonne Paramètres
                        with gr.Column(elem_classes=["studio-card"]):
                            gr.HTML("<div class='card-heading'>Paramètres Vocaux</div>")

                            tts_voice_type = gr.Radio(
                                choices=["Voix enregistrée", "Fichier personnalisé"],
                                value="Voix enregistrée" if initial_voices else "Fichier personnalisé",
                                label="Source de référence",
                            )
                            tts_preset_dropdown = gr.Dropdown(
                                choices=initial_voices,
                                value=initial_voices[0] if initial_voices else None,
                                label="Profil vocal",
                                visible=bool(initial_voices),
                            )
                            tts_custom_audio = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="Extrait audio de référence (3-10 sec)",
                                visible=not bool(initial_voices),
                            )

                            def toggle_tts_input(choice):
                                if choice == "Voix enregistrée":
                                    return gr.update(visible=True), gr.update(visible=False)
                                return gr.update(visible=False), gr.update(visible=True)

                            tts_voice_type.change(
                                toggle_tts_input,
                                inputs=[tts_voice_type],
                                outputs=[tts_preset_dropdown, tts_custom_audio],
                            )

                            tts_language = gr.Dropdown(
                                choices=languages,
                                value="French" if "French" in languages else "English",
                                label="Langue de synthèse",
                            )

                            with gr.Accordion("Paramètres Avancés", open=False):
                                tts_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Vitesse d'élocution")
                                tts_temp = gr.Slider(0.1, 1.0, value=0.75, step=0.05, label="Expressivité")
                                tts_rep_penalty = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Pénalité de répétition")

                        # Colonne Script & Rendu
                        with gr.Column(elem_classes=["studio-card"]):
                            gr.HTML("<div class='card-heading'>Script & Synthèse</div>")

                            tts_text_input = gr.Textbox(
                                lines=5,
                                placeholder="Saisissez ou collez le texte à synthétiser...",
                                label="Texte à générer",
                                value="Bienvenue sur VoiceFlow Studio. Votre texte est converti avec fidélité en une voix fluide et naturelle.",
                            )

                            tts_btn = gr.Button("Générer la voix", variant="primary", elem_classes=["primary-btn"])
                            tts_status_text = gr.Markdown("", elem_classes=["status-pill-result"])

                            tts_output_audio = gr.Audio(
                                label="Rendu Audio",
                                type="filepath",
                                waveform_options=gr.WaveformOptions(
                                    waveform_color="#7C3AED",
                                    waveform_progress_color="#4F46E5",
                                ),
                            )

                    tts_btn.click(
                        handle_tts,
                        inputs=[
                            tts_voice_type,
                            tts_preset_dropdown,
                            tts_custom_audio,
                            tts_text_input,
                            tts_language,
                            tts_speed,
                            tts_temp,
                            tts_rep_penalty,
                        ],
                        outputs=[tts_output_audio, tts_status_text],
                    )

                # ================= TAB 2: CHANGEMENT DE VOIX =================
                with gr.TabItem("Changement de Voix"):
                    with gr.Row(elem_classes=["studio-layout-grid"]):

                        with gr.Column(elem_classes=["studio-card"]):
                            gr.HTML("<div class='card-heading'>Source Audio & Transcription</div>")
                            sts_source_audio = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="Audio source",
                            )
                            sts_source_language = gr.Dropdown(
                                choices=["Auto-detect", *languages],
                                value="Auto-detect",
                                label="Langue de l'enregistrement",
                            )
                            sts_transcribe_btn = gr.Button("Transcrire l'audio", variant="secondary", elem_classes=["secondary-btn"])
                            sts_transcription_info = gr.Markdown("", elem_classes=["status-pill-result"])

                            sts_text_input = gr.Textbox(
                                lines=3,
                                placeholder="Texte transcrit...",
                                label="Transcription",
                            )

                        with gr.Column(elem_classes=["studio-card"]):
                            gr.HTML("<div class='card-heading'>Transformation Vocale</div>")
                            with gr.Row():
                                sts_preset_dropdown = gr.Dropdown(
                                    choices=initial_voices,
                                    value=initial_voices[0] if initial_voices else None,
                                    label="Voix cible",
                                )
                                sts_language = gr.Dropdown(
                                    choices=languages,
                                    value="French" if "French" in languages else "English",
                                    label="Langue de sortie",
                                )

                            with gr.Accordion("Paramètres Avancés", open=False):
                                sts_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Vitesse d'élocution")
                                sts_temp = gr.Slider(0.1, 1.0, value=0.75, step=0.05, label="Expressivité")
                                sts_rep_penalty = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Pénalité de répétition")

                            sts_transform_btn = gr.Button("Transformer la voix", variant="primary", elem_classes=["primary-btn"])
                            sts_status_text = gr.Markdown("", elem_classes=["status-pill-result"])

                            with gr.Row():
                                sts_original_player = gr.Audio(label="Voix d'origine", type="filepath")
                                sts_output_player = gr.Audio(
                                    label="Voix transformée",
                                    type="filepath",
                                    waveform_options=gr.WaveformOptions(
                                        waveform_color="#7C3AED",
                                        waveform_progress_color="#4F46E5",
                                    ),
                                )

                    sts_transcribe_btn.click(
                        handle_transcription,
                        inputs=[sts_source_audio, sts_source_language],
                        outputs=[sts_text_input, sts_transcription_info],
                    )

                    sts_transform_btn.click(
                        handle_speech_to_speech,
                        inputs=[
                            sts_source_audio,
                            sts_text_input,
                            tts_voice_type,
                            sts_preset_dropdown,
                            tts_custom_audio,
                            sts_language,
                            sts_speed,
                            sts_temp,
                            sts_rep_penalty,
                        ],
                        outputs=[sts_original_player, sts_output_player, sts_status_text],
                    )

                # ================= TAB 3: BIBLIOTHÈQUE DE VOIX =================
                with gr.TabItem("Bibliothèque"):
                    with gr.Row(elem_classes=["studio-layout-grid"]):

                        with gr.Column(elem_classes=["studio-card"]):
                            gr.HTML("<div class='card-heading'>Enrôlement Vocal</div>")
                            lib_name_input = gr.Textbox(placeholder="Ex: Emma, Marc, Narrateur...", label="Nom du profil vocal")
                            lib_audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Extrait audio (3 à 10 secondes)")
                            lib_save_btn = gr.Button("Enregistrer la voix", variant="primary", elem_classes=["primary-btn"])
                            lib_status = gr.Markdown("", elem_classes=["status-pill-result"])

                        with gr.Column(elem_classes=["studio-card"]):
                            gr.HTML("<div class='card-heading'>Recommandations Techniques</div>")
                            gr.Markdown(
                                """
                                **Critères d'enregistrement optimal :**
                                
                                - **Durée recommandée** : 3 à 10 secondes d'élocution continue et claire.
                                - **Environnement acoustique** : Sans réverbération, musique de fond ni souffle.
                                - **Ponctuation** : Le modèle XTTS s'appuie sur la ponctuation pour structurer les pauses et le rythme naturel.
                                """
                            )

                    lib_save_btn.click(
                        handle_save_voice,
                        inputs=[lib_audio_input, lib_name_input],
                        outputs=[lib_status, tts_preset_dropdown, sts_preset_dropdown],
                    )

    return demo, theme

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def get_preferred_port():
    port_value = os.environ.get("GRADIO_SERVER_PORT", "7860")
    try:
        port = int(port_value)
    except ValueError:
        return get_free_port()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return port
        except OSError:
            fallback_port = get_free_port()
            print(f"Port {port} is already in use. Starting on port {fallback_port}.")
            return fallback_port

if __name__ == "__main__":
    app, theme = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=get_preferred_port(),
        share=False,
        theme=theme,
    )
