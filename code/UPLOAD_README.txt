Musculoskeletal Robotic Shoulder Bioreactor - source folder

EDIT AND UPLOAD
The application source of truth is this code folder.
Entry point: qt_gui.py
GUI and controllers: control/
Web UI and embedded CAD: pam_shoulder_gui.html
Physics models: shoulder_pam_model.py and shoulder_pam_model_v6.py
Force/MPC weights and configuration: control/ and control/models/
Firmware: arduino test/, arduino_force/, giga-regulator-control/
Nano25 reader and calibration: setup_for_shuai/
Existing sensor-test folders are preserved.
Nano25 manuals are in ../file/Nano25_manuals, not runtime source.
Optional hardware installers are in ../data/installation_media/Nano25.
Historical LLM files are archived in ../data/output/history_llm.
The retained September 27 request reports in control are required by Load 1/2;
do not remove them merely because they look like old requests.

START
Use Start_Bioreactor.bat, or Python 3.10 to run qt_gui.py.
On a new computer, run Setup_Environment.bat first; Internet is required
unless a suitable offline runtime and wheel cache is supplied separately.
Check_Environment.bat performs an offline software check without robot control.
The existing ../data/.python310 environment is reused when available.

DATA
Default output stays outside code, in the sibling data folder:
  output/recordings/     GUI, response and camera recordings
  output/control_logs/  controller and calibration logs
  nano25_recordings/    standalone Nano25 recordings
Existing GUI-selected save folders remain in effect.
Set TRO_DATA_ROOT to override the default data root.

UPLOAD SCOPE
Include all tracked source, HTML, model weights, configuration and calibration.
Do not upload only Python files. Do not upload .python310, runtime caches,
__pycache__, .pio, historical logs/videos, model backups or credentials.
.gitignore excludes these from normal Git staging; a raw folder copy does
not apply .gitignore. Use Pack_Code.ps1 for a filtered upload ZIP instead.

COMPATIBILITY
Launch from code/qt_gui.py or code/Start_Bioreactor.bat.
The project root also has qt_gui.py and Start_Bioreactor.bat convenience shims;
both delegate to code and contain no controller implementation.
Obsolete outer launch-only shims are archived in file/legacy_root_launchers;
they are not runnable entry points in the new layout.
Run firmware upload separately through Arduino IDE/PlatformIO when required.
Changing Python code does not automatically update firmware on the robot.
