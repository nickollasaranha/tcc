import Main
import sys

CHAR_ASPECT_RATIO_INTERVAL = [float(x)/float(10) for x in range(11)]
PLATE_ASPECT_RATIO_INTERVAL = [float(x)/float(10) for x in range(20, 101, 5)]

min_char_aspect_ratio = float(sys.argv[1])

# Plate Aspect Ratio
for max_char_aspect_ratio in CHAR_ASPECT_RATIO_INTERVAL:
    if min_char_aspect_ratio >= max_char_aspect_ratio: continue

    for min_plate_aspect_ratio in PLATE_ASPECT_RATIO_INTERVAL:
        for max_plate_aspect_ratio in PLATE_ASPECT_RATIO_INTERVAL:
            if min_plate_aspect_ratio>=max_plate_aspect_ratio: continue
            Main.run([min_char_aspect_ratio, max_char_aspect_ratio], [min_plate_aspect_ratio, max_plate_aspect_ratio])