# Copyright 2026 Bob Ros
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import queue
import re
import subprocess
import tempfile
import threading
import time

from ament_index_python.packages import get_package_share_directory
import numpy as np

from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from std_msgs.msg import Bool, Int16MultiArray, String


"""Fish Audio S2-Pro TTS ROS 2 Node."""


class TTSnode(Node):
    """ROS node that provides an interface to Fish Audio S2-Pro with streaming text aggregation."""

    def __init__(self):
        """Initialize the TTS node and its parameters."""
        super().__init__('tts')

        # Get package share directory for defaults
        try:
            package_share = get_package_share_directory('bob_s2ptts')
        except Exception:
            package_share = ''

        # 1. ROS Parameters - Core
        self.declare_parameter(
            'model_id',
            os.environ.get('S2PTTS_MODEL_ID', 'fishaudio/s2-pro'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description=('Fish Audio model id (HF hub) or absolute path to a '
                             'local checkpoint directory.')
            )
        )
        self.declare_parameter(
            'model_dir',
            os.environ.get('S2PTTS_MODEL_DIR', os.path.join(os.getcwd(), 'models')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Local directory where models are stored/cached.'
            )
        )
        self.declare_parameter(
            'sentence_delimiters',
            (os.environ.get('S2PTTS_SENTENCE_DELIMITERS', '').split(',')
             if os.environ.get('S2PTTS_SENTENCE_DELIMITERS', '') else ['. ', '! ', '? ']),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=('Delimiters that trigger sentence aggregation. '
                             'Use [""] to disable splitting.')
            )
        )
        self.declare_parameter(
            'flush_timeout',
            int(os.environ.get('S2PTTS_FLUSH_TIMEOUT', '700')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Timeout in ms to flush the text buffer without a delimiter.'
            )
        )

        # 1. ROS Parameters - Generation (S2-Pro sampling)
        # NOTE: S2-Pro uses repetition-aware sampling; repetition_penalty and
        # do_sample from the Qwen node have no equivalent and are not declared.
        self.declare_parameter(
            'temperature',
            float(os.environ.get('S2PTTS_TEMPERATURE', '0.7')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Sampling temperature (schema bounds: 0.1..1.0).'
            )
        )
        self.declare_parameter(
            'top_p',
            float(os.environ.get('S2PTTS_TOP_P', '0.7')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Top-p sampling (schema bounds: 0.1..1.0).'
            )
        )
        self.declare_parameter(
            'top_k',
            int(os.environ.get('S2PTTS_TOP_K', '30')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Top-k sampling (>=0).'
            )
        )
        self.declare_parameter(
            'chunk_length',
            int(os.environ.get('S2PTTS_CHUNK_LENGTH', '200')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description=('Iterative-prompt chunk length in tokens '
                             '(schema bounds: 100..1000).')
            )
        )
        self.declare_parameter(
            'max_new_tokens',
            int(os.environ.get('S2PTTS_MAX_NEW_TOKENS', '1024')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Max audio frames per chunk (0 = model context cap).'
            )
        )
        self.declare_parameter(
            'seed',
            int(os.environ.get('S2PTTS_SEED', '-1')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Fixed random seed (-1 for random).'
            )
        )
        self.declare_parameter(
            'normalize',
            os.environ.get('S2PTTS_NORMALIZE', 'true').lower() == 'true',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='Normalize text for number stability.'
            )
        )

        # 1. ROS Parameters - Runtime
        self.declare_parameter(
            'precision',
            os.environ.get('S2PTTS_PRECISION', 'bfloat16'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Model precision: "bfloat16" or "float16".'
            )
        )
        self.declare_parameter(
            'compile',
            os.environ.get('S2PTTS_COMPILE', 'false').lower() == 'true',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='torch.compile the backbone (slow start, faster inference).'
            )
        )
        self.declare_parameter(
            'warmup',
            os.environ.get('S2PTTS_WARMUP', 'true').lower() == 'true',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='Run a dummy synthesis after loading.'
            )
        )

        # 1. ROS Parameters - Voice Clone
        # S2-Pro has no separate instruction channel: control tags like
        # [whisper] are inline in the text. control_instructions is prefixed.
        self.declare_parameter(
            'control_instructions',
            os.environ.get('S2PTTS_CONTROL_INSTRUCTIONS', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description=('Fish inline control tags (e.g. "[whisper]") '
                             'prepended to the generated text. Empty = off.')
            )
        )
        self.declare_parameter(
            'voice_ref_audio',
            os.environ.get(
                'S2PTTS_VOICE_REF_AUDIO',
                os.path.join(package_share, 'config', 'eva_44khz_de.wav')
            ),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Reference audio file for voice cloning. Empty = random timbre.'
            )
        )
        self.declare_parameter(
            'voice_ref_text',
            os.environ.get(
                'S2PTTS_VOICE_REF_TEXT',
                os.path.join(package_share, 'config', 'voice_ref_text_de.txt')
            ),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Transcript or path to transcript text for reference audio.'
            )
        )

        # 1. ROS Parameters - Output
        self.declare_parameter(
            'play',
            os.environ.get('S2PTTS_PLAY', 'true').lower() == 'true',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='Whether to play the audio immediately.'
            )
        )
        self.declare_parameter(
            'player',
            os.environ.get('S2PTTS_PLAYER', 'sys'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Audio player: "sys" or executable path (e.g. aplay).'
            )
        )
        self.declare_parameter(
            'audio_device',
            os.environ.get('S2PTTS_AUDIO_DEVICE', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Device ID or name for sounddevice playback.'
            )
        )
        self.declare_parameter(
            'file_prefix',
            os.environ.get('S2PTTS_FILE_PREFIX', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Prefix for saved audio files.'
            )
        )
        self.declare_parameter(
            'target_sample_rate',
            int(os.environ.get('S2PTTS_TARGET_SAMPLE_RATE', '0')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Force resampling to this rate (0 for auto).'
            )
        )
        self.declare_parameter(
            'file_start_index',
            int(os.environ.get('S2PTTS_FILE_START_INDEX', '1')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Starting index for saved audio files.'
            )
        )
        self.declare_parameter(
            'speaking_flag_delay',
            float(os.environ.get('S2PTTS_SPEAKING_FLAG_DELAY', '0.2')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Delay in seconds before setting speaking_flag to false.'
            )
        )
        self.declare_parameter(
            'sentence_pause',
            float(os.environ.get('S2PTTS_SENTENCE_PAUSE', '0.6')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Silence inserted after each played sentence (seconds).'
            )
        )
        self.declare_parameter(
            'fade_ms',
            int(os.environ.get('S2PTTS_FADE_MS', '8')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description=('Fade-in/out per utterance in ms. Prevents clicks from '
                             'playback stream start/stop between sentences. 0 disables.')
            )
        )
        # 2. State & Queues
        self.text_buffer = ''
        self.last_text_time = self.get_clock().now()
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.lock = threading.Lock()
        self.declare_parameter(
            'substitute',
            [''],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=('Array of [regex, replacement] pairs for text cleaning. '
                             'Use [""] for no replacement.')
            )
        )
        self.file_index = self.get_parameter('file_start_index').value
        self.inferred_target_sr = None
        self._is_speaking = False
        self._speaking_finish_time = 0.0  # Time in seconds (float)

        # 3. Subscriber & Publisher
        self.sub = self.create_subscription(
            String,
            'text',
            self.text_callback,
            1000
        )
        self.pub = self.create_publisher(
            String,
            'spoken_text',
            10
        )
        self.audio_pub = self.create_publisher(
            Int16MultiArray,
            'audio_raw',
            10
        )

        # 4. Speaking Flag (Transient Local / Latched)
        speaking_qos = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.speaking_pub = self.create_publisher(
            Bool,
            'speaking_flag',
            speaking_qos
        )
        # Initialize flag to False
        self.speaking_pub.publish(Bool(data=False))

        # 4. Initialization & Threading
        self.engine = None
        self.sd = None
        self.sf = None
        self.torch = None
        self.get_logger().info('Loading Fish Audio S2-Pro libraries and model...')

        try:
            # Set environment variables before model loading.
            # HF_HOME alone: modern huggingface_hub stores under <model_dir>/hub/.
            # (Do NOT set HUGGINGFACE_HUB_CACHE to model_dir here: that legacy
            # override breaks the hub-cache layout of huggingface_hub >= 1.0.)
            model_dir = self.get_parameter('model_dir').value
            os.environ['HF_HOME'] = model_dir

            import torch
            self.torch = torch

            precision = self.get_parameter('precision').value.lower()
            if precision not in ('bfloat16', 'float16'):
                self.get_logger().error(
                    f"Invalid precision '{precision}'. Use 'bfloat16' or 'float16'."
                )
                return
            dtype = torch.bfloat16 if precision == 'bfloat16' else torch.float16
            compile_model = self.get_parameter('compile').value

            # Resolve checkpoint: absolute dir or HF hub id cached in model_dir
            checkpoint_path = self._resolve_checkpoint(
                self.get_parameter('model_id').value, model_dir
            )

            # Try importing sounddevice for native playback
            try:
                import sounddevice as sd
                self.sd = sd
            except ImportError:
                self.sd = None

            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.dac.inference import load_model as load_decoder_model
            from fish_speech.models.text2semantic.inference import (
                launch_thread_safe_queue,
                resolve_codec_path,
            )

            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device='cuda:0',
                precision=dtype,
                compile=compile_model,
            )
            decoder_model = load_decoder_model(
                config_name='modded_dac_vq',
                checkpoint_path=resolve_codec_path(checkpoint_path, None),
                device='cuda:0',
            )
            self.engine = TTSInferenceEngine(
                llama_queue=self.llama_queue,
                decoder_model=decoder_model,
                precision=dtype,
                compile=compile_model,
                model_family='s2',
            )

            if self.get_parameter('warmup').value:
                self.get_logger().info('Warming up model...')
                self._synthesize('Warmup.', warmup=True)
                self.get_logger().info('Warmup done.')

            self.get_logger().info('Model loaded successfully.')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize TTS node: {e}')
            return

        self.tts_thread = threading.Thread(target=self.tts_loop, daemon=True)
        self.audio_thread = threading.Thread(target=self.audio_loop, daemon=True)

        self.tts_thread.start()
        self.audio_thread.start()

        # Timers
        self.flush_timer = self.create_timer(0.1, self.flush_timer_callback)
        # 20Hz monitor
        self.speaking_monitor_timer = self.create_timer(0.05, self.speaking_monitor_callback)

        self.get_logger().info("TTS Node ready and listening on topic 'text'.")

    def _resolve_checkpoint(self, model_id, model_dir):
        """Return a local checkpoint dir: use model_id path or HF hub cache in model_dir."""
        if os.path.isdir(model_id):
            return model_id
        try:
            # HF_HOME was set to model_dir before import; prefer the local cache
            from huggingface_hub import snapshot_download
            try:
                return snapshot_download(repo_id=model_id, local_files_only=True)
            except Exception:
                self.get_logger().info('Checkpoint not cached locally, downloading...')
                return snapshot_download(repo_id=model_id)
        except Exception as e:
            raise RuntimeError(
                f'Could not resolve checkpoint "{model_id}" (dir or HF hub): {e}'
            ) from e

    def _get_sf(self):
        """Lazy load soundfile library."""
        if self.sf is None:
            try:
                import soundfile as sf
                self.sf = sf
            except ImportError as e:
                self.get_logger().error(f'Failed to import soundfile: {e}')
                raise e
        return self.sf

    def text_callback(self, msg):
        """Aggregate incoming words/text into sentences based on delimiters."""
        delimiters = self.get_parameter('sentence_delimiters').value
        input_text = msg.data

        with self.lock:
            self.last_text_time = self.get_clock().now()
            self.text_buffer += input_text

            # Bypass aggregation completely if delimiters list is empty
            if not delimiters or (len(delimiters) == 1 and delimiters[0] == ''):
                # If we have something like [''], it will be handled by flush_timeout
                return

            # Process buffer for any of the delimiters using regex for robustness
            while True:
                earliest_match = None

                for d in delimiters:
                    if not d:
                        continue

                    try:
                        # Match the delimiter literally but allow any whitespace
                        # if it ends with a space.
                        pattern = d
                        if d.endswith(' '):
                            pattern = re.escape(d.rstrip()) + r'\s+'

                        match = re.search(pattern, self.text_buffer)
                        if match:
                            if earliest_match is None or match.start() < earliest_match.start():
                                earliest_match = match
                    except re.error as e:
                        self.get_logger().error(f"Invalid delimiter regex '{d}': {e}")
                        continue

                if earliest_match is None:
                    break

                # Extract sentence up to and including the matched delimiter
                split_point = earliest_match.end()
                sentence = self.text_buffer[:split_point].strip()
                self.text_buffer = self.text_buffer[split_point:]

                if sentence:
                    # Apply regex filters
                    sentence = self._apply_substitutions(sentence)
                    if sentence:
                        self.get_logger().info(f"Aggregated sentence: '{sentence}'")
                        self.text_queue.put(sentence)

    def flush_timer_callback(self):
        """Flush the text buffer if the timeout has passed."""
        timeout_ms = self.get_parameter('flush_timeout').value
        if timeout_ms <= 0:
            return

        with self.lock:
            if not self.text_buffer:
                return

            now = self.get_clock().now()
            diff = (now - self.last_text_time).nanoseconds / 1e6
            if diff > timeout_ms:
                sentence = self.text_buffer.strip()
                if sentence:
                    # Apply regex filters
                    sentence = self._apply_substitutions(sentence)
                    if sentence:
                        self.get_logger().info(f"Aggregated sentence: '{sentence}'")
                        self.text_queue.put(sentence)
                self.text_buffer = ''

    def _apply_substitutions(self, text):
        """Apply regex-based substitutions to the text."""
        substitutes = self.get_parameter('substitute').value
        if not substitutes or (len(substitutes) == 1 and substitutes[0] == ''):
            return text

        # Input must be pairs of [regex, replacement]
        if len(substitutes) % 2 != 0:
            self.get_logger().warn('Substitute parameter has odd length. Last item ignored.')
            substitutes = substitutes[:-1]

        for i in range(0, len(substitutes), 2):
            pattern = substitutes[i]
            replacement = substitutes[i+1]
            try:
                text = re.sub(pattern, replacement, text)
            except re.error as e:
                self.get_logger().error(f"Regex error for pattern '{pattern}': {e}")

        return text.strip()

    def _synthesize(self, text, warmup=False):
        """Run one S2-Pro synthesis. Returns (audio ndarray, sample_rate)."""
        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

        references = []
        if not warmup:
            ref_audio = self.get_parameter('voice_ref_audio').value
            ref_text_param = self.get_parameter('voice_ref_text').value
            if ref_audio and os.path.isfile(ref_audio):
                # Resolve ref_text if it's a file path
                ref_text = ref_text_param
                if ref_text and (os.path.isfile(ref_text) or ref_text.startswith('/')):
                    try:
                        if os.path.isfile(ref_text):
                            with open(ref_text, 'r', encoding='utf-8') as f:
                                ref_text = f.read().strip()
                    except Exception as e:
                        self.get_logger().error(f'Failed to read ref_text {ref_text_param}: {e}')
                        ref_text = ''
                with open(ref_audio, 'rb') as f:
                    references.append(
                        ServeReferenceAudio(audio=f.read(), text=ref_text or '')
                    )
            elif ref_audio:
                self.get_logger().warn(f'Reference audio not found: {ref_audio}')

        # S2-Pro control tags are inline; prepend them instead of a separate channel
        control_instructions = '' if warmup else self.get_parameter('control_instructions').value
        full_text = f'{control_instructions} {text}'.strip() if control_instructions else text

        # Clamp to ServeTTSRequest schema bounds to get clear errors early
        temperature = min(max(self.get_parameter('temperature').value, 0.1), 1.0)
        top_p = min(max(self.get_parameter('top_p').value, 0.1), 1.0)
        top_k = max(self.get_parameter('top_k').value, 0)
        chunk_length = min(max(self.get_parameter('chunk_length').value, 100), 1000)
        max_new_tokens = max(self.get_parameter('max_new_tokens').value, 0)
        seed = self.get_parameter('seed').value
        normalize = self.get_parameter('normalize').value

        req = ServeTTSRequest(
            text=full_text,
            chunk_length=chunk_length,
            references=references,
            use_memory_cache='on',
            normalize=normalize,
            streaming=False,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            seed=None if seed < 0 else seed,
        )

        audio, sr = None, None
        for result in self.engine.inference(req):
            if result.code == 'error':
                raise RuntimeError(f'S2-Pro inference error: {result.error}')
            if result.code == 'final':
                sr, audio = result.audio
        if audio is None:
            raise RuntimeError('No audio generated.')
        return audio, sr

    def tts_loop(self):
        """Process the text queue and generate audio."""
        while rclpy.ok():
            try:
                # Wait for text with a timeout to allow checking rclpy.ok()
                text = self.text_queue.get(timeout=1.0)

                self.get_logger().info(f"Generating audio for: '{text}'")
                start_time = time.time()

                audio, sr = self._synthesize(text)

                latency = time.time() - start_time
                self.get_logger().info(f'Generated audio in {latency:.2f}s')

                # Push to audio queue (with text for feedback)
                self.audio_queue.put((audio, sr, text))

                # Cleanup cache
                if self.torch and self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()

                self.text_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in TTS loop: {e}')

    def audio_loop(self):
        """Consume the audio queue for playback and storage."""
        while rclpy.ok():
            try:
                audio_data, sr, spoken_text = self.audio_queue.get(timeout=1.0)

                # 0. Fade edges to avoid click transients at stream start/stop
                fade_ms = self.get_parameter('fade_ms').value
                if fade_ms > 0:
                    audio_data = self._apply_fade(audio_data, sr, fade_ms)

                # 1. Save to file if prefix is provided
                file_prefix = self.get_parameter('file_prefix').value
                if file_prefix:
                    filename = f'{file_prefix}_{self.file_index:05d}.wav'
                    self._get_sf().write(filename, audio_data, sr)
                    self.get_logger().info(f'Saved audio to {filename}')
                    self.file_index += 1

                # 2. Publish text before starting playback
                msg = String()
                msg.data = spoken_text
                self.pub.publish(msg)
                self.get_logger().info(f"Speaking: '{spoken_text}'")

                # 3. Update Speaking Flag state
                duration = len(audio_data) / sr
                with self.lock:
                    now_seconds = self.get_clock().now().nanoseconds / 1e9
                    # Start of processing this chunk
                    self._speaking_finish_time = max(
                        self._speaking_finish_time, now_seconds) + duration

                    if not self._is_speaking:
                        self._is_speaking = True
                        self.speaking_pub.publish(Bool(data=True))
                        self.get_logger().debug('Speaking started (speaking_flag -> True)')

                # 4. Stream raw audio if subscribed (Bridge for streamer)
                # NOTE: .tolist() and resampling can be VERY slow for long audio!
                if self.audio_pub.get_subscription_count() > 0:
                    try:
                        t_bridge_start = time.time()
                        stream_audio = self.resample_audio(audio_data, sr, 44100)
                        stream_int16 = (stream_audio * 32767).astype(np.int16)

                        audio_msg = Int16MultiArray()
                        audio_msg.data = stream_int16.flatten().tolist()
                        self.audio_pub.publish(audio_msg)

                        bridge_dur = time.time() - t_bridge_start
                        if bridge_dur > 0.5:
                            self.get_logger().warn(
                                f'Audio bridge processing took {bridge_dur:.2f}s!')
                    except Exception as e:
                        self.get_logger().error(f'Failed to stream raw audio: {e}')

                # 5. Play if enabled
                if self.get_parameter('play').value:
                    self.play_audio(audio_data, sr)
                    # Human-like gap before the next sentence
                    pause = self.get_parameter('sentence_pause').value
                    if pause > 0:
                        time.sleep(pause)
                    # After physical playback finished, update the timestamp to the actual NOW
                    with self.lock:
                        actual_now = self.get_clock().now().nanoseconds / 1e9
                        self._speaking_finish_time = actual_now
                        self.get_logger().debug(
                            'Playback finished, flag will reset in '
                            f'{self.get_parameter("speaking_flag_delay").value}s')

                self.audio_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in audio loop: {e}')

    def _apply_fade(self, audio_data, sr, fade_ms):
        """Apply short linear fade-in/out to prevent edge clicks."""
        n = min(int(sr * fade_ms / 1000), len(audio_data) // 2)
        if n <= 0:
            return audio_data
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        audio_data = audio_data.astype(np.float32, copy=True)
        audio_data[:n] *= ramp
        audio_data[-n:] *= ramp[::-1]
        return audio_data

    def play_audio(self, audio_data, sr):
        """Play audio using native library or configured executable."""
        player = self.get_parameter('player').value

        if player == 'sys':
            # Use Python-native library (sounddevice)
            if self.sd is not None:
                # 1. Resolve target sample rate
                # (explicit param > inferred working rate > model native)
                target_sr = self.get_parameter('target_sample_rate').value
                if target_sr <= 0 and self.inferred_target_sr:
                    target_sr = self.inferred_target_sr

                if target_sr > 0 and target_sr != sr:
                    audio_data = self.resample_audio(audio_data, sr, target_sr)
                    sr = target_sr

                # 2. Attempt playback
                try:
                    self._do_native_play(audio_data, sr)
                except Exception as e:
                    # If auto-resample failed or was invalid, try 48000 (standard for HDMI)
                    if 'Invalid sample rate' in str(e) and sr != 48000:
                        self.get_logger().warn(
                            f'Playback at {sr}Hz failed. Retrying at 48000Hz fallback.'
                        )
                        resampled_data = self.resample_audio(audio_data, sr, 48000)
                        try:
                            self._do_native_play(resampled_data, 48000)
                            # If fallback succeeded, remember this for next time
                            self.inferred_target_sr = 48000
                            self.get_logger().info('Sticky sample rate set to 48000Hz.')
                            return
                        except Exception as e2:
                            self.get_logger().error(f'Retry at 48000Hz failed: {e2}')

                    self.get_logger().warn(f'Native playback failed: {e}. Falling back to aplay.')
                    self._play_with_executable('aplay', audio_data, sr)
            else:
                self.get_logger().warn('sounddevice not installed. Falling back to aplay.')
                self._play_with_executable('aplay', audio_data, sr)
        else:
            # Use configured external player
            self._play_with_executable(player, audio_data, sr)

    def _do_native_play(self, audio_data, sr):
        """Perform native audio playback using sounddevice."""
        device = self.get_parameter('audio_device').value
        if device:
            try:
                device = int(device)
            except ValueError:
                pass
            self.sd.play(audio_data, sr, device=device)
        else:
            self.sd.play(audio_data, sr)
        self.sd.wait()

    def resample_audio(self, audio_data, orig_sr, target_sr):
        """Resample audio data using librosa if available, else linear interpolation."""
        try:
            import librosa
            return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            self.get_logger().warn('librosa not installed. Using simple numpy interpolation.')
            duration = len(audio_data) / orig_sr
            num_samples = int(duration * target_sr)
            return np.interp(
                np.linspace(0, len(audio_data), num_samples),
                np.arange(len(audio_data)),
                audio_data
            ).astype(audio_data.dtype)

    def _play_with_executable(self, executable, audio_data, sr):
        """Play audio using an external command."""
        temp_file_path = None
        try:
            # Use a standard python library for cross-platform temporary files
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as ntf:
                temp_file_path = ntf.name

            # Write audio to the temporary file
            self._get_sf().write(temp_file_path, audio_data, sr)

            # Run the external player
            subprocess.run(
                [executable, temp_file_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            self.get_logger().error(f'Playback with {executable} failed: {e}')
        finally:
            # Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    self.get_logger().warn(f'Failed to remove temp file {temp_file_path}: {e}')

    def speaking_monitor_callback(self):
        """Monitor the speaking duration and reset flag after delay."""
        if not self._is_speaking:
            return

        now_seconds = self.get_clock().now().nanoseconds / 1e9
        delay = self.get_parameter('speaking_flag_delay').value

        with self.lock:
            diff = (self._speaking_finish_time + delay) - now_seconds
            if diff < 0:
                self._is_speaking = False
                self.speaking_pub.publish(Bool(data=False))
                self._speaking_finish_time = 0.0
                self.get_logger().info('Speaking finished (speaking_flag -> False)')
            elif diff < 2.0:  # Only log if close to finishing, to avoid spam
                self.get_logger().debug(f'Speaking flag still active. Reset in {diff:.2f}s')


def main(args=None):
    """Start the TTS node."""
    import argparse
    parser = argparse.ArgumentParser(
        description='ROS node that provides an interface to Fish Audio S2-Pro '
                    'with streaming text aggregation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-l', '--list', action='store_true', help='list available audodevices')
    parsed, remaining = parser.parse_known_args()

    if parsed.list:
        import sounddevice as sd
        print(sd.query_devices())
        parser.exit(0)

    rclpy.init(args=args)
    node = TTSnode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down node...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
