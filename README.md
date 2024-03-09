# Security-voice-code-access

## Overview:

The Security Voice-code Access software component provides robust access control based on voice recognition technology. With two operation modes, users can choose between a specific pass-code sentence or individual voice fingerprints for secure access. Developed using fingerprint and spectrogram concepts, this software ensures reliable access control for up to 8 users.

## Features:

### Mode 1 - Security Voice Code:

- Access is granted only upon speaking a specific pass-code sentence.
- Valid passcodes include "Open middle door", "Unlock the gate", and "Grant me access".

### Mode 2 - Security Voice Fingerprint:

- Access is granted based on individual voice fingerprints.
- Users can select which of the original 8 users are granted access.

### Settings UI:

- Users can configure access settings, including selecting individuals granted access in Mode 2.

### Voice Recording Button:

- Initiates voice recording for the pass-code sentence.

### Spectrogram Viewer:

- Displays the spectrogram of the spoken voice-code for analysis.

### Analysis Results Summary:

- Provides a summary of analysis results, including:
  - A table showing the match percentage of the spoken sentence with each saved passcode sentence.
  - A table showing the match percentage of the spoken voice with each of the 8 saved individuals.

### Access Status Indicator:

- UI element indicating whether access is granted or denied based on algorithm results.


## Libraries Used:

- **Python**: Programming language used for development.
- **NumPy**: For numerical operations and array manipulations.
- **Matplotlib**: For creating visualizations, including spectrogram display.
- **SciPy**: For scientific computing functions, including signal processing.
- **SoundFile**: For reading and writing audio files.

## Preview:

![Screenshot 1](Task%205/screenshots/1.png)
![Screenshot 2](Task%205/screenshots/2.png)
![Screenshot 3](Task%205/screenshots/3.png)
![Screenshot 4](Task%205/screenshots/4.png)



## Contributors <a name = "Contributors"></a>

<table>
  <tr>
    <td align="center">
      <div style="text-align:center; margin-right:20px;">
        <a href="https://github.com/OmarEmad101">
          <img src="https://github.com/OmarEmad101.png" width="100px" alt="@OmarEmad101">
          <br>
          <sub><b>Omar Emad</b></sub>
        </a>
      </div>
    </td>
    <td align="center">
      <div style="text-align:center; margin-right:20px;">
        <a href="https://github.com/Omarnbl">
          <img src="https://github.com/Omarnbl.png" width="100px" alt="@Omarnbl">
          <br>
          <sub><b>Omar Nabil</b></sub>
        </a>
      </div>
    </td>
    <td align="center">
      <div style="text-align:center; margin-right:20px;">
        <a href="https://github.com/KhaledBadr07">
          <img src="https://github.com/KhaledBadr07.png" width="100px" alt="@KhaledBadr07">
          <br>
          <sub><b>Khaled Badr</b></sub>
        </a>
      </div>
    </td>
    <td align="center">
      <div style="text-align:center; margin-right:20px;">
        <a href="https://github.com/merna-abdelmoez">
          <img src="https://github.com/merna-abdelmoez.png" width="100px" alt="@merna-abdelmoez">
          <br>
          <sub><b>Mirna Abdelmoez</b></sub>
        </a>
      </div>
    </td>
  </tr>
</table>

## Acknowledgments

**This project was supervised by Dr. Tamer Basha & Eng. Abdallah Darwish, who provided invaluable guidance and expertise throughout its development as a part of the Digital Signal Processing course at Cairo University Faculty of Engineering.**

<div style="text-align: right">
    <img src="https://imgur.com/Wk4nR0m.png" alt="Cairo University Logo" width="100" style="border-radius: 50%;"/>
</div>
