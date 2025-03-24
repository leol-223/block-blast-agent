import subprocess

applescript = """
tell application "System Events"
    tell process "iPhone Mirroring"
        set position of window 1 to {666, 95}
    end tell
end tell
"""

subprocess.run(["osascript", "-e", applescript])