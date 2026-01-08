import mediapipe
print("Dir mediapipe:", dir(mediapipe))
try:
    import mediapipe.python.solutions as solutions
    print("Found mediapipe.python.solutions")
except ImportError as e:
    print("Error importing mediapipe.python.solutions:", e)

try:
    from mediapipe import solutions
    print("Found from mediapipe import solutions")
except ImportError as e:
    print("Error from mediapipe import solutions:", e)
