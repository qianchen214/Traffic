# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matlab
import matlab.engine

eng = matlab.engine.start_matlab()
x1 = 149
y1 = 122
x2 = 5
y2 = 245
signal = eng.localization_refinement('11.jpg',matlab.double([x1]),matlab.double([y1]),matlab.double([x2]),matlab.double([y2]),'red',nargout=4)
print(signal)
