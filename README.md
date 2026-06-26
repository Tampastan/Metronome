打包exe：
pyinstaller --noconfirm --onefile --windowed --icon=icon.ico --name "Metronome" --add-data "icon.ico;." --collect-all _sounddevice_data metronome.py
