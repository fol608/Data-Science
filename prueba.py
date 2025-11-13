import sounddevice as sd
print(sd.query_devices())
print("Default:", sd.default.device)