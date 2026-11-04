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
import signal
import sys

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea, QSlider,
    QVBoxLayout, QWidget
)
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
import rclpy
from rclpy.node import Node


class GUInode(Node, QMainWindow):
    """ROS 2 node with a Qt GUI for tuning TTS parameters."""

    def __init__(self):
        """Initialize the GUI node and its UI components."""
        # Initialize ROS node
        Node.__init__(self, 'tts_gui')
        # Initialize Qt Window
        QMainWindow.__init__(self)

        self.setWindowTitle('S2-Pro TTS Parameter Tuner')
        self.setMinimumWidth(1000)
        self.setMinimumHeight(850)

        self.target_node = os.environ.get('S2GUI_TARGET', 'eva/tts/tts')
        self.set_cli = self.create_client(
            SetParameters, f'/{self.target_node}/set_parameters'
        )
        self.get_cli = self.create_client(
            GetParameters, f'/{self.target_node}/get_parameters'
        )

        # Setup UI
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Header with Refresh Button
        self.header = QHBoxLayout()
        self.refresh_btn = QPushButton('Sync from Node')
        self.refresh_btn.clicked.connect(self.sync_parameters)
        self.header.addWidget(QLabel(f'Target: /{self.target_node}'))
        self.header.addStretch()
        self.header.addWidget(self.refresh_btn)
        self.main_layout.addLayout(self.header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.scroll.setWidget(self.content_widget)
        self.main_layout.addWidget(self.scroll)

        # Parameter definitions
        self.params = [
            # Area: Generation
            {'name': 'temperature', 'type': 'double', 'min': 0.1, 'max': 1.0,
             'scale': 100, 'label': 'Temperature', 'group': 'Generation'},
            {'name': 'top_p', 'type': 'double', 'min': 0.1, 'max': 1.0,
             'scale': 100, 'label': 'Top-P', 'group': 'Generation'},
            {'name': 'top_k', 'type': 'int', 'min': 0, 'max': 100,
             'label': 'Top-K (0=off)', 'group': 'Generation'},
            {'name': 'chunk_length', 'type': 'int', 'min': 100, 'max': 1000,
             'label': 'Chunk Length', 'group': 'Generation'},
            {'name': 'max_new_tokens', 'type': 'int', 'min': 0, 'max': 2048,
             'label': 'Max New Tokens (0=model cap)', 'group': 'Generation'},
            {'name': 'seed', 'type': 'int', 'min': -1, 'max': 999999,
             'label': 'Seed (-1=random)', 'group': 'Generation'},

            # Area: Runtime
            {'name': 'compile', 'type': 'bool', 'group': 'Runtime',
             'label': 'torch.compile (restart required)'},
            {'name': 'normalize', 'type': 'bool', 'group': 'Runtime',
             'label': 'Normalize Text'},

            # Area: Voice Clone
            {'name': 'control_instructions', 'type': 'string',
             'label': 'Control Tags (e.g. [whisper])', 'group': 'Voice Clone'},
            {'name': 'voice_ref_audio', 'type': 'string',
             'label': 'Voice Ref Audio (.wav)', 'group': 'Voice Clone',
             'browse': True},
            {'name': 'voice_ref_text', 'type': 'string',
             'label': 'Voice Ref Text (ICL)', 'group': 'Voice Clone'},

            # Area: Output & Storage
            {'name': 'play', 'type': 'bool', 'label': 'Enable Playback',
             'group': 'Output & Storage'},
            {'name': 'player', 'type': 'string', 'label': 'Player (aplay/sys)',
             'group': 'Output & Storage'},
            {'name': 'audio_device', 'type': 'string', 'label': 'Audio Device (ID/Name)',
             'group': 'Output & Storage'},
            {'name': 'target_sample_rate', 'type': 'int', 'min': 0, 'max': 96000,
             'label': 'Target SR (0=auto)', 'group': 'Output & Storage'},
            {'name': 'file_prefix', 'type': 'string', 'label': 'File Save Prefix',
             'group': 'Output & Storage'},
            {'name': 'sentence_pause', 'type': 'double', 'min': 0.0, 'max': 2.0,
             'scale': 100, 'label': 'Pause after Sentence (s)',
             'group': 'Output & Storage'},
            {'name': 'fade_ms', 'type': 'int', 'min': 0, 'max': 50,
             'label': 'Edge Fade (ms)', 'group': 'Output & Storage'},
            {'name': 'file_start_index', 'type': 'int', 'min': 1, 'max': 99999,
             'label': 'File Start Index', 'group': 'Output & Storage'},
            {'name': 'speaking_flag_delay', 'type': 'double', 'min': 0.0, 'max': 2.0,
             'scale': 100, 'label': 'Flag Delay after Speech (s)',
             'group': 'Output & Storage'},

            # Area: Pipeline Logic
            {'name': 'flush_timeout', 'type': 'int', 'min': 0, 'max': 5000,
             'label': 'Flush Timeout (ms)', 'group': 'Pipeline Logic'},
        ]

        self.groups = {}
        self.widgets = {}
        self.value_labels = {}

        self.setup_parameters()

        # Timer for ROS 2 spin_once (20 Hz)
        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self.ros_loop)
        self.ros_timer.start(50)

        # Initial Sync
        QTimer.singleShot(1000, self.sync_parameters)

    def ros_loop(self):
        """Perform ROS 2 spinning in the main Qt thread."""
        if rclpy.ok():
            try:
                rclpy.spin_once(self, timeout_sec=0)
            except Exception:
                pass

    def setup_parameters(self):
        """Create GUI widgets for each parameter using a Grid Layout for flexibility."""
        for p in self.params:
            group_name = p.get('group', 'Other')
            if group_name not in self.groups:
                box = QGroupBox(group_name)
                # Using GridLayout for better alignment and resizability
                grid = QGridLayout()
                grid.setColumnStretch(1, 1)  # Slider/Input column is flexible
                box.setLayout(grid)
                self.content_layout.addWidget(box)
                self.groups[group_name] = [grid, 0]  # Store layout and next row index

            layout, row = self.groups[group_name]

            # Label (fixed width to prevent cutoff)
            label = QLabel(p['label'])
            label.setFixedWidth(280)
            layout.addWidget(label, row, 0)

            if p['type'] == 'bool':
                widget = QCheckBox()
                widget.stateChanged.connect(
                    lambda state, name=p['name']: self.on_bool_change(name, state)
                )
                layout.addWidget(widget, row, 1)
            elif p['type'] in ['int', 'double']:
                widget = QSlider(Qt.Horizontal)
                widget.setMinimum(int(p['min'] * p.get('scale', 1)))
                widget.setMaximum(int(p['max'] * p.get('scale', 1)))

                value_label = QLabel('N/A')
                value_label.setFixedWidth(60)
                self.value_labels[p['name']] = value_label

                if p['type'] == 'double':
                    widget.valueChanged.connect(
                        lambda val, name=p['name'], sc=p['scale'], lbl=value_label:
                        self.on_double_change(name, val, sc, lbl)
                    )
                else:
                    widget.valueChanged.connect(
                        lambda val, name=p['name'], lbl=value_label:
                        self.on_int_change(name, val, lbl)
                    )

                layout.addWidget(widget, row, 1)
                layout.addWidget(value_label, row, 2)
            elif p['type'] == 'string':
                widget = QLineEdit()
                widget.editingFinished.connect(
                    lambda name=p['name'], w=widget:
                    self.on_string_change(name, w.text())
                )

                if p.get('browse'):
                    container = QWidget()
                    h_layout = QHBoxLayout(container)
                    h_layout.setContentsMargins(0, 0, 0, 0)
                    h_layout.addWidget(widget)

                    browse_btn = QPushButton('...')
                    browse_btn.setFixedWidth(40)
                    browse_btn.clicked.connect(
                        lambda checked, w=widget, name=p['name']:
                        self.browse_file(w, name)
                    )
                    h_layout.addWidget(browse_btn)
                    layout.addWidget(container, row, 1)
                else:
                    layout.addWidget(widget, row, 1)

            self.widgets[p['name']] = widget
            self.groups[group_name][1] += 1  # Increment row counter

    def browse_file(self, line_edit, param_name):
        """Open a file dialog to select a WAV file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Reference Audio', '',
            'Audio Files (*.wav);;All Files (*)', options=options
        )
        if file_path:
            line_edit.setText(file_path)
            self.on_string_change(param_name, file_path)

    def sync_parameters(self):
        """Read all current values from the target node."""
        if not self.get_cli.service_is_ready():
            return

        req = GetParameters.Request()
        req.names = [p['name'] for p in self.params]

        future = self.get_cli.call_async(req)
        future.add_done_callback(self.on_sync_response)

    def on_sync_response(self, future):
        """Update GUI widgets based on current ROS parameter values."""
        try:
            res = future.result()
            for i, p_val in enumerate(res.values):
                name = self.params[i]['name']
                p_type_name = self.params[i]['type']
                widget = self.widgets[name]

                # Block signals during sync to avoid feedback loop
                widget.blockSignals(True)

                if p_type_name == 'bool':
                    widget.setChecked(p_val.bool_value)
                elif p_type_name == 'int':
                    val = p_val.integer_value
                    widget.setValue(val)
                    if name in self.value_labels:
                        self.value_labels[name].setText(str(val))
                elif p_type_name == 'double':
                    val = p_val.double_value
                    scale = next(item['scale'] for item in self.params
                                 if item['name'] == name)
                    widget.setValue(int(val * scale))
                    if name in self.value_labels:
                        self.value_labels[name].setText(f'{val:.2f}')
                elif p_type_name == 'string':
                    widget.setText(p_val.string_value)

                widget.blockSignals(False)
            self.get_logger().info('Successfully synced parameters from node.')
        except Exception as e:
            self.get_logger().error(f'Failed to sync parameters: {e}')

    def on_bool_change(self, name, state):
        """Update a boolean parameter."""
        val = (state == Qt.Checked)
        self.set_ros_param(name, ParameterType.PARAMETER_BOOL, val)

    def on_int_change(self, name, val, label):
        """Update an integer parameter and its label."""
        label.setText(str(val))
        self.set_ros_param(name, ParameterType.PARAMETER_INTEGER, int(val))

    def on_double_change(self, name, val, scale, label):
        """Update a double parameter and its label."""
        real_val = val / float(scale)
        label.setText(f'{real_val:.2f}')
        self.set_ros_param(name, ParameterType.PARAMETER_DOUBLE, float(real_val))

    def on_string_change(self, name, val):
        """Update a string parameter."""
        self.set_ros_param(name, ParameterType.PARAMETER_STRING, str(val))

    def set_ros_param(self, name, p_type, val):
        """Send a SetParameters request to the target node."""
        if not self.set_cli.service_is_ready():
            return

        param = Parameter()
        param.name = name
        p_val = ParameterValue()
        if p_type == ParameterType.PARAMETER_BOOL:
            p_val.bool_value = val
        elif p_type == ParameterType.PARAMETER_INTEGER:
            p_val.integer_value = val
        elif p_type == ParameterType.PARAMETER_DOUBLE:
            p_val.double_value = val
        elif p_type == ParameterType.PARAMETER_STRING:
            p_val.string_value = val

        param.value = p_val

        req = SetParameters.Request()
        req.parameters = [param]
        self.set_cli.call_async(req)


def main(args=None):
    """Start the GUI node."""
    # Ensure Ctrl+C terminates the application immediately
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    rclpy.init(args=args)

    app = QApplication(sys.argv)
    node = GUInode()
    node.show()

    try:
        sys.exit(app.exec_())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
