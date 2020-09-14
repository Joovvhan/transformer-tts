# https://programmers.co.kr/learn/courses/21/lessons/950
# https://financedata.github.io/posts/matplotlib-hangul-for-ubuntu-linux.html

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

print (f"{'Version.':15}: ", matplotlib.__version__)
print (f"{'Location':15}: ", matplotlib.__file__)
print (f"{'Configs':15}: ", matplotlib.get_configdir())
print (f"{'Cache':15}: ", matplotlib.get_cachedir())
print()

font_list = [f.fname for f in matplotlib.font_manager.fontManager.ttflist]
for font in (font_list):
    print(font)
print()

print (f"Current Font size ({plt.rcParams['font.size']}) / Current Font Family {plt.rcParams['font.family']}")
print()

plt.rcParams["font.family"] = 'Nanum Gothic'

print (f"Current Font size ({plt.rcParams['font.size']}) / Current Font Family {plt.rcParams['font.family']}")
print()

font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
for font in (font_list):
    print(font)
print()

ttf_font_list = [f.name for f in fm.fontManager.ttflist]
for font in (ttf_font_list):
    print(font)
print()