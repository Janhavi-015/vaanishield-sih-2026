# VaaniShield

**A self-monitoring, dual-microphone neural denoiser for defence communication, designed to run in real time on a Raspberry Pi 4.**

Smart India Hackathon 2026 | Problem Statement **SIH26052** | Category: Hardware | Theme: Miscellaneous
Team **6 Solvers** | Team ID **186200**

> **PROJECT STATUS: DESIGN STAGE.** This repository documents the design, dataset specification and evaluation plan. Every item is labelled *Designed*, *Planned*, *In progress* or *Implemented* in the Status table (Section 11). **Nothing in this README is a measured result unless it says so. Targets are targets, not achievements.**

---

## Table of contents

1. [The problem](#1-the-problem)
2. [Why classical noise cancellation fails here](#2-why-classical-noise-cancellation-fails-here)
3. [Our idea in one paragraph](#3-our-idea-in-one-paragraph)
4. [System architecture](#4-system-architecture)
5. [Block-by-block explainer](#5-block-by-block-explainer)
6. [Dataset pipeline](#6-dataset-pipeline)
7. [The neural model and training](#7-the-neural-model-and-training)
8. [Hardware prototype](#8-hardware-prototype)
9. [Safety design](#9-safety-design)
10. [Evaluation plan](#10-evaluation-plan)
11. [Project status](#11-project-status)
12. [What we claim and what we do not](#12-what-we-claim-and-what-we-do-not)
13. [Known limitations and risks](#13-known-limitations-and-risks)
14. [Repository layout](#14-repository-layout)
15. [Getting started (planned)](#15-getting-started-planned)
16. [Impact and future scope](#16-impact-and-future-scope)
17. [Team](#17-team)
18. [References](#18-references)
19. [License and disclaimer](#19-license-and-disclaimer)

---

## 1. The problem

PS SIH26052: develop an AI/ML-enabled adaptive noise cancellation (ANC) system that suppresses **stationary, non-stationary and impulsive** defence noises while maintaining **high speech intelligibility** and **real-time performance on embedded hardware**.

A soldier's radio or headset audio is buried under:

| Noise type | Examples | Why it is hard |
|---|---|---|
| Stationary | Engine hum, wind, rotor drone | Steady, but often as loud as speech |
| Non-stationary | Vehicle passing, crowd, machinery changes | Statistics change faster than simple filters adapt |
| Impulsive | Gunshots, artillery, blasts | Last only a few milliseconds but are extremely loud |

Misheard commands are a mission risk, and very loud impulses are also a hearing-safety risk for the listener.

**Our interpretation of the PS.** We treat this as noise cancellation on the *communication path* (microphone to headset audio), i.e. speech enhancement. We do not produce physical anti-noise inside the earcup in this phase (see Future scope).

## 2. Why classical noise cancellation fails here

- **Spectral subtraction and Wiener filtering** assume the noise is roughly stationary so it can be estimated in speech pauses. Engines and wind fit this; gunshots and changing vehicle noise do not.
- **LMS/NLMS adaptive filters** need a reference signal correlated with the noise and can diverge, or cancel the wearer's own voice, if they keep adapting during speech or impulses.
- **Frame-based neural models** typically work on 20 ms STFT frames. A gunshot lasts roughly 1-5 ms, so it is shorter than one frame and gets smeared across it instead of being removed cleanly.
- **Single-model systems fail silently.** If a microphone is blocked or the model is too slow, most systems just produce bad or no audio.

Our design attacks each of these failure modes with a specific block (Section 5).

## 3. Our idea in one paragraph

VaaniShield is a wearable, real-time speech-enhancement system. Two microphones feed a Raspberry Pi 4, where a **classical guard**, an **adaptive filter** and a **neural denoiser** clean the audio, and a **safety limiter** protects the listener's ears before it goes to the headset. The neural model (fine-tuned DeepFilterNet3) does the heavy lifting; the small classical blocks around it handle the cases where a neural model alone is known to be weak, and the self-monitoring blocks make the system degrade safely instead of going silent.

The base model is **DeepFilterNet3**. Our contribution is the surrounding pipeline, the dataset design, the dual-microphone conditioning and the evaluation plan. **Every additional block is a hypothesis that our ablation will test; blocks that show no measured gain are dropped before the demo.**

## 4. System architecture

Three layers, matching the project presentation:

| Layer | Name | Blocks |
|---|---|---|
| 1 | Sense and Guard | Dual mics, impulse detector, speech gate, mic-health monitor |
| 2 | Classical | Guarded NLMS removes noise both mics hear |
| 3 | Neural | Fine-tuned DeepFilterNet3-based dual-mic model removes the remaining stationary, periodic and impulsive noise |

Around these: confidence blend, soft limiter, watchdog fallback and a live dashboard.

### Live pipeline (Raspberry Pi 4)

```
Mic1 + Mic2 (I2S, same clock)
   -> Capture thread
        |-> Mic-health monitor (rule-based)
   -> Impulse guard + speech gate
   -> Guarded NLMS (step size set by the gate)
   -> STFT (2 mics + inter-mic ratio)
   -> DFN3 dual-mic model (ONNX int8, FiLM-conditioned)
   -> Confidence blend
   -> iSTFT (overlap-add)
   -> Soft limiter (-3 dBFS ceiling)
   -> USB DAC -> Headset (volume-limited)

Watchdog: if the model is slow or fails -> NLMS + limiter -> pass-through + limiter
```

### Offline pipeline (laptop / Colab)

```
Defence noise + speech -> 2-mic simulator (data mixer, SNR mix)
   -> Train / fine-tune (DFN3 + losses)
   -> Export and save (ONNX int8)
   -> model.onnx + config -> SD card of the Pi
```

### Graceful degradation ladder

| Level | Path | When it is used |
|---|---|---|
| Full | Guard, NLMS, DFN3 core, confidence blend, limiter | Normal operation |
| Reduced | NLMS + limiter | Model too slow or failed |
| Minimal | Pass-through + limiter | Everything else failed |

## 5. Block-by-block explainer

Each block lists what it does, how, and which failure it addresses.

### 5.1 Layer 1: Sense and Guard

**Impulse detector.** A frame-energy jump of roughly 15 dB over a running median flags a gunshot or blast. A short-frame (1-2 ms) time-domain detector is used because a gunshot is shorter than one STFT frame. Thresholds are starting values, to be tuned on the prototype.

**Speech gate.** The wearer-speech flag comes from a high Mic1/Mic2 energy ratio, because Mic 1 is near the mouth and Mic 2 faces away.

**Purpose of both:** stop the adaptive filter from blowing up on gunshots or cancelling the wearer's own voice, and tell the neural model when an impulse is present.

**Mic-health monitor (no neural network).** Simple, explainable signal checks on raw microphone frames:

- Blocked or dead mic: energy collapse in one channel
- Wind: strong low-frequency energy with low inter-mic coherence (wind is uncorrelated between mics)
- Clipping: samples at full scale
- Touch or handling thump: low-frequency burst in one mic only
- Dropout or clock issue: missing or repeated blocks

Output: a health flag and a trust weight per microphone. It is deliberately independent of the neural model, so it still works when the model is wrong. If it is built, we report detection rate and false-alarm rate on injected faults (not SNR).

### 5.2 Layer 2: Guarded NLMS

A normalised LMS adaptive filter that uses Mic 2 as a noise reference for Mic 1, removing noise that both microphones hear (engine, rotor).

| Condition | Step size mu | Effect |
|---|---|---|
| Impulse (gunshot / blast) | 0 | Weights frozen; the model and limiter handle the blast |
| Wearer speech present | 0 | NLMS cannot learn to cancel the voice |
| New noise onset | High (start 0.5-0.8) | Filter adapts quickly |
| Steady noise | Low (start about 0.05) | Fine tracking, no jitter |

These values are starting guesses and must be tuned on the real prototype. NLMS stays in the final system **only if the ablation shows a gain**.

### 5.3 Layer 3: Neural core (DeepFilterNet3, dual-mic, FiLM-conditioned)

- **Base:** pretrained DeepFilterNet3 (ERB-band gains plus deep-filter complex taps, complex-domain so phase is preserved).
- **Dual-mic input:** the reference microphone is given to the model as side information (complex STFT plus inter-mic magnitude ratio and phase difference).
- **Impulse-aware conditioning:** the 1-2 ms detector flag and the speech-gate flag feed the network through FiLM layers (scale and shift on encoder features).
- **Speech-preservation rule:** where speech probability is high, per-bin attenuation is capped (starting point 12 dB, to be tuned on validation data), trading a little residual noise for intelligibility.

### 5.4 Confidence blend

```
out = c * enhanced + (1 - c) * conservative
```

`conservative` is a mild-gain version of Mic 1. The confidence `c` comes from the model's per-bin confidence and, where possible, agreement with a classical noise-estimate gain. If predicted output quality is below the input's, the output falls back toward pass-through.

Wording: this is a *confidence estimate, not a guarantee*, until we show calibration.

### 5.5 Soft limiter and watchdog

- **Look-ahead soft limiter**, ceiling about -3 dBFS, protects the listener from peaks and from any blast residue the model lets through.
- **Watchdog** monitors processing time per frame and model health, and steps down the ladder in Section 4. The demo never goes silent.

### 5.6 Live dashboard

A small view of mic health, confidence, real-time factor and latency. It is a debugging and demo aid, **not a claim of performance**.

## 6. Dataset pipeline

An automated two-microphone simulation builds every training example:

```
Mic 1 = room(speech) + g * room(noise)
Mic 2 = leak * room(speech) + g * room(noise, different delays)    # leak about -15 to -25 dB
Target = dry speech at Mic 1
```

| Parameter | Planned range |
|---|---|
| SNR | -5 to 15 dB (training); evaluation reported at 0, 5, 10 dB |
| Events per clip | 1 to 3 impulsive events, some deliberately on syllables |
| Reverb | Simulated room impulse responses (pyroomacoustics) |
| Two-mic simulation | Mic 1 near mouth, Mic 2 facing away, speech leakage into Mic 2 |
| Augmentation | Random gain, EQ, extra reverb, radio band-limit, hard clipping, reference-mic dropout |
| Labels saved | SNR, noise class (stationary / periodic / impulsive), impulse onset times |
| Splits | Speaker-disjoint **and** noise-file-disjoint; some noise types fully held out for testing |

**Data sources:** LibriSpeech and DNS Challenge speech and noise, a range-recorded gunshot dataset, C3GD, plus our own recordings on the prototype microphones (test only). Operating rate is 48 kHz because pretrained DeepFilterNet3 is native 48 kHz; 16 kHz-only corpora are upsampled or skipped.

**Consistency rule.** Whatever runs live before the model (guard, NLMS) must also run inside the data pipeline. Otherwise the model trains on data that differs from what it sees in deployment. If NLMS is not applied in training, the model is fed raw mics and NLMS is kept only as a separate output path.

**Honest limitation.** This is synthetic data built from public clips, not recordings from a real defence source. We compensate with held-out noise types and real recordings on our prototype. The pipeline is designed to accept real DRDO-provided noise recordings later without changes.

## 7. The neural model and training

**Approach: fine-tune DeepFilterNet3, do not train from scratch.**

Staged plan (each stage leaves a working fallback):

| Plan | System | Role |
|---|---|---|
| C (demo-safe) | Pretrained DFN3 + limiter | Guarantees a live demo; built first |
| B | Single-channel fine-tuned DFN3 + guarded NLMS + limiter | Upgrade on top of C |
| A (full) | Dual-mic + impulse/noise-type conditioning + guarded NLMS + limiter | Default target if time allows |

**Losses (planned):** SI-SNR, L1/L2 spectral losses, multi-resolution STFT and a perceptual loss, with extra weight on frames containing impulses.

**Hyper-parameter search:** Optuna on a data subset (a few tens of short trials).

**Deployment:** export to ONNX and quantise to int8 for the Pi. If the measured real-time factor (RTF) is too high, a lighter model (GTCRN) is the fallback.

**Reference benchmarks (third-party, laptop CPUs, not ours):** DeepFilterNet reports RTF 0.04 (DFN2) and 0.19 (DFN3). **The Raspberry Pi 4 benchmark is pending**; measuring it is our Day 1 milestone.

**Licence of the base model:** DeepFilterNet3 is released under MIT/Apache-2.0. Confirm against the repository before redistributing weights.

## 8. Hardware prototype

| Item | Choice |
|---|---|
| Compute | Raspberry Pi 4 (4 GB) with heatsink, fan and 5V/3A supply |
| Microphones | 2x INMP441 I2S MEMS on one I2S bus (same clock, so sample-synchronised) |
| Playback | USB DAC + headset (volume-limited) |
| Backup | ReSpeaker 2-Mic HAT, spare SD card with identical image |
| Optional | Jetson (TensorRT) comparison, since the PS mentions Jetson AGX Orin "or similar" |

**Wiring (INMP441 to Raspberry Pi I2S):** VDD to 3.3 V, GND to GND, SCK to GPIO18, WS to GPIO19, SD to GPIO20. Mic 1 (near the mouth) has its L/R pin to GND; Mic 2 (facing away) has L/R to 3.3 V, so they share the bus as left and right channels.

**Mic placement matters.** Mic 1 on a short boom near the mouth with a foam windscreen; Mic 2 on the outside of the headset, facing away, port unblocked.

**Real-time budget:** the Pi 4 CPU budget is unmeasured. Measuring RTF and end-to-end latency is the first milestone, and it decides which model (DFN3 or GTCRN) and which optional blocks we actually run.

## 9. Safety design

- **Ear safety:** look-ahead soft limiter at about -3 dBFS.
- **Fail-safe audio:** the watchdog degrades the pipeline rather than muting it.
- **Safe testing:** noise and gunshot recordings are played from a speaker at safe levels. We never play real gunshot levels into a headset. The limiter is verified before volume is raised.
- **No silent failure:** microphone faults and model failures are flagged and shown on the dashboard.

## 10. Evaluation plan

We keep the evaluation focused on what the PS asks for.

| Metric | What it tells us | PS requirement |
|---|---|---|
| Output SI-SNR (and SNR improvement) | Noise reduction | Suppress noise |
| STOI | Objective speech intelligibility | High intelligibility |
| RTF, end-to-end latency, CPU load on the Pi | Embedded feasibility | Real-time on embedded hardware |
| PESQ (optional) | Perceptual speech quality | Supporting metric |

SI-SNR and STOI are reported **per noise type** (stationary, non-stationary, impulsive) and **per input SNR**, using held-out noise types not seen in training.

### Ablation study (values are filled in only after measurement)

| ID | System | Output SI-SNR | STOI | RTF / latency on Pi |
|---|---|---|---|---|
| A0 | Noisy input (baseline) | Not measured yet | Not measured yet | n/a |
| A1 | Pretrained DFN3 | Not measured yet | Not measured yet | Not measured yet |
| A2 | + fine-tune on defence-like noise | Not measured yet | Not measured yet | Not measured yet |
| A3 | + reference mic (dual-mic) | Not measured yet | Not measured yet | Not measured yet |
| A4 | + guarded NLMS | Not measured yet | Not measured yet | Not measured yet |

Any row that does not show a measured gain is dropped from the final system. Mic-health monitor and confidence blend get their own check if built (detection / false-alarm rate on injected faults; calibration plot for confidence), not an SNR row.

### Safety checks (simple tests, not quality scores)

- **Limiter:** feed a loud impulse and confirm the output peak stays under the ceiling.
- **Watchdog:** stop or slow the model and confirm the pipeline falls back without going silent.
- **Gate:** a clap freezes NLMS weights; the wearer talking is not cancelled.

### Targets, not results

We aim for roughly **15 dB output SNR, 0.85 STOI** and real-time operation on the Pi (the PS also lists PESQ above 2.5). **These are targets. None has been measured or achieved.** Gunshots and very low input SNR (-5 dB) are expected to be harder, which is why we report per condition rather than one average.

## 11. Project status

| Component | Status |
|---|---|
| Problem analysis and architecture design | Designed (documented here) |
| Dataset specification (2-mic simulation) | Designed |
| Dataset generator script | Planned |
| Pretrained DFN3 + limiter + watchdog live demo | Planned (first milestone) |
| Real-time benchmark on Raspberry Pi 4 | Planned |
| Fine-tuning (single-channel, then dual-mic) | Planned |
| Guard (impulse + speech gate) and guarded NLMS | Planned |
| Mic-health monitor | Planned |
| Confidence blend | Planned |
| Live dashboard | Planned |
| Evaluation on the core ablation | Not started |

*(Update this table as work lands: Planned -> In progress -> Implemented.)*

## 12. What we claim and what we do not

**We can say**

- A complete, PS-aligned pipeline design: dataset, ERB sub-band + full-band complex-domain neural model, SI-SNR/L1/L2/perceptual losses, optional guarded LMS, ONNX int8 edge deployment, and a live microphone-to-headset prototype plan.
- A safety-oriented design: self-monitoring, graceful degradation and ear protection.

**We do not say**

- That the SNR, STOI or PESQ targets are achieved.
- That any added block improves results. Each is a hypothesis for the ablation to test.
- That the base model is ours. It is DeepFilterNet3; our additions are the blocks above.
- That the system works in the field or is certified for operational use.
- That a block is built unless the Status table says so.

## 13. Known limitations and risks

| Risk | Mitigation |
|---|---|
| Raspberry Pi 4 CPU budget unmeasured | Benchmark first (Day 1); ONNX int8; GTCRN as lighter fallback |
| Gunshots (1-5 ms) shorter than a 20 ms STFT frame | Time-domain detector + FiLM conditioning + impulse-weighted loss + limiter |
| NLMS weights blow up on a blast or cancel the voice | mu = 0 on impulse and wearer speech; normalised update |
| Speech gate misfires (e.g. wind gust at Mic 1) | Tune on real mics, smooth over frames; model also sees raw mics |
| Train/test mismatch after adding NLMS | Run the same gate + NLMS in the data pipeline, or drop NLMS (ablation decides) |
| Few defence-specific public noise clips | Held-out noise types; pipeline accepts real recordings later |
| Simulation-to-real gap | EQ/RIR augmentation and a light fine-tune on real recordings |
| Mic clips at gunshot level (INMP441 overload point about 120 dB SPL, per datasheet) | Clipping augmentation; demo with recorded clips at safe level; a limiter cannot repair a clipped input |
| int8 quantisation hurts quality | Compare fp32 vs int8; keep sensitive layers in higher precision |
| Clock drift between I2S mics and USB DAC | ALSA/PipeWire resampling; monitor buffer level |
| Python audio glitches | Separate I/O and inference threads; larger buffer; C++/Rust runtime if needed |
| I2S overlay or OS mismatch | Test on Day 1; ReSpeaker HAT backup path; passthrough stage first |
| Confidence estimate may be over-confident | Calibration plot; call it an estimate, not a guarantee |
| Hardware fails during demo | Spare Pi/SD image, pre-recorded before/after audio, watchdog fallback, rehearsal |

## 14. Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── docs/
│   ├── architecture.md          # block diagram and signal flow
│   ├── dataset_spec.md          # 2-mic mixing, SNR ranges, augmentations
│   ├── evaluation_plan.md       # metrics, protocol, ablation table
│   ├── hardware_setup.md        # wiring, I2S configuration, BOM
│   └── images/                  # diagrams and charts
├── data/
│   └── README.md                # how to obtain datasets; no large files in git
├── src/
│   ├── dataset/                 # mixer, 2-mic simulation
│   ├── blocks/                  # mic-health monitor, guard, nlms, confidence blend, limiter
│   ├── model/                   # DFN3 wrapper, dual-mic and FiLM extensions
│   ├── runtime/                 # live pipeline, watchdog, dashboard
│   └── eval/                    # SI-SNR, STOI, PESQ, RTF/latency scripts
├── scripts/
│   ├── make_dataset.py
│   ├── train.py
│   ├── export_onnx.py
│   ├── benchmark_rtf.py
│   └── run_live.py
└── results/
    └── README.md                # only measured numbers, with the command that produced them
```

## 15. Getting started (planned)

These commands describe the intended workflow. Scripts are added as they are implemented; **check the Status table before running anything.**

```bash
git clone https://github.com/Janhavi-015/vaanishield-sih-2026.git
cd vaanishield-sih-2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Generate synthetic 2-mic data (planned)
python scripts/make_dataset.py --out data/synth --snr-min -5 --snr-max 15

# 2) Measure real-time factor on the target board (planned, first milestone)
python scripts/benchmark_rtf.py --model pretrained_dfn3

# 3) Live demo, Tier 0: pretrained model + limiter + watchdog (planned)
python scripts/run_live.py --tier 0
```

**Software stack:** Python, NumPy/SciPy, PyTorch (training), ONNX Runtime (edge inference), pyroomacoustics (simulation), pystoi and pesq (metrics), Optuna (hyper-parameter search).

## 16. Impact and future scope

**Intended impact (to be validated by the evaluation above)**

- **Soldiers and infantry:** aims to make radio commands clearer amid gunfire, artillery and vehicle noise.
- **Vehicle and aircrew:** aims to cut rotor, engine and wind noise on the headset path.
- **Disaster response (NDRF/SDRF):** aims to keep communication usable near machinery, sirens and collapsed structures.
- **Industrial high-noise sites:** named in the PS as a deployment sector beyond defence.
- **Cost and reuse:** a Pi-class device could replace custom DSP boards (exact bill of materials to be confirmed); new noise types plug into the same dataset and training pipeline.
- **Strategic:** an indigenous, deployable speech-enhancement pipeline for defence communication, aligned with DRDO/iDEX and Atmanirbhar Bharat, designed to accept real DRDO noise recordings without redesign.

**Future scope**

- Phase 2, physical ANC: FxLMS-based anti-noise inside the earcup, complementing the communication-path system.
- Real defence data: plug in DRDO-provided noise recordings in place of public datasets.
- Further extensions, each with its own ablation row if built: time-domain impulse repair, a three-stream (speech / continuous / impulsive) separator.
- Confidence calibration and distillation into a smaller model.
- Gunshot direction estimation from inter-mic delay (coarse with closely spaced microphones).

## 17. Team

**Team name:** 6 Solvers | **Team ID:** 186200 | **Problem Statement:** SIH26052

| S. No. | Name | Role |
|---|---|---|
| 1 | Rudraksh Budhappa | Leader |
| 2 | Ritesh Ranjan | Member |
| 3 | Sagar | Member |
| 4 | Isha Thakur | Member |
| 5 | Sarwat Hussian Sheik | Member |
| 6 | Janhavi Tidke | Member |

## 18. References

> Verify authors, years and venues before citing in a final report.

- Narain et al., DRDO, 2026: review of noise-cancellation / speech-enhancement work for military noise.
- Gurny and Quinn, 2026: gunshot data and short-impulse characteristics.
- Akarsh et al.; Dai Shuning (NTU, 2026): deep ANC and speech-preserving active noise control vs FxLMS.
- Schroter et al.: DeepFilterNet family (DeepFilterNet, DeepFilterNet2, DeepFilterNet3).
- Westhausen and Meyer: DTLN and DTLN-aec.
- GTCRN: lightweight speech-enhancement model (fallback candidate); IVA + GTCRN for dual-mic systems.
- Datasets: LibriSpeech, DNS Challenge noise, range-recorded gunshot dataset, C3GD.
- pyroomacoustics (room and microphone simulation); pystoi and pesq (metrics).
- INMP441 (TDK) datasheet; NIOSH 1998 criteria (85 dBA 8-hour limit; impulse noise not above 140 dB peak); MIL-STD-1474D/E.

## 19. License and disclaimer

**Code:** MIT License (see `LICENSE`). Third-party models and datasets keep their own licences.

**Disclaimer:** this is a hackathon prototype. It has not been tested or certified for operational, safety-critical or field use. Do not rely on it for hearing protection or mission communication.
