Musculoskeletal Robotic Shoulder Bioreactor - Version 0.1
Portable Windows x64 package

FIRST USE
1. Extract the entire ZIP. Do not run the program from inside the ZIP.
2. Keep at least 3 GB free on the destination drive.
3. In code, double-click Setup_Environment.bat once. A source-only upload
   downloads Python and dependencies unless a suitable offline cache is supplied.
4. Double-click Start_Bioreactor.bat.

NORMAL USE
- Start_Bioreactor.bat starts both the Qt GUI and its local MPC service.
- Check_Environment.bat validates Python, GUI libraries, MLP-MPC models,
  configuration files, FFmpeg, and recording write access without touching hardware.
- Select the serial port inside the GUI. Validation never connects to the robot.
- Camera access is requested only after opening Data Visualization and the camera.
- Recordings are saved under ..\data\output\recordings unless another path is selected.

LLM + MPC
- GPT account mode requires Codex CLI login on the destination computer.
- Claude account mode requires Claude Code login on the destination computer.
- API keys are entered at runtime and are not included in this ZIP.
- The real-time 20 Hz loop remains MPC; the LLM only designs bounded compensation
  between completed runs.

CONTROL MODEL
- The packaged default is the tested LTV solver using the MLP forward model.
- acados generated sources are included for development, but acados itself is
  machine-specific and is not required by the packaged default configuration.

HARDWARE/FIRMWARE
- Arduino GIGA task firmware source: arduino test\teach_mode\src\teach_mode\teach_mode.ino
  Legacy regulator firmware is retained under giga-regulator-control.
- A different computer may still require the Arduino GIGA USB/serial driver.
- Always confirm the correct COM port and use the GUI emergency stop before real tests.

PACKAGE INTEGRITY
- For a filtered source ZIP, run Pack_Code.ps1.
- Run Check_Environment.bat after copying or extracting the package.

PACKAGE SCOPE
- The code folder keeps application source, control/model files, embedded CAD
  in the HTML, controller firmware, and environment setup scripts.
- Historical logs/videos, papers, document-generation sources, model backups,
  individual-part STL files, and development tests remain in the original project;
  they are not required to operate the software on another computer.
