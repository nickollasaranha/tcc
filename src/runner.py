import Main
import sys

CHAR_ASPECT_RATIO_INTERVAL = [float(x)/float(10) for x in range(11)]
PLATE_ASPECT_RATIO_INTERVAL = [float(x)/float(10) for x in range(20, 101, 5)]

start = int(sys.argv[1])
combs = []
# Plate Aspect Ratio
for min_plate_aspect_ratio in PLATE_ASPECT_RATIO_INTERVAL:
  for max_plate_aspect_ratio in PLATE_ASPECT_RATIO_INTERVAL:
    if min_plate_aspect_ratio>=max_plate_aspect_ratio: continue
    if min_plate_aspect_ratio<=4: continue
    combs.append([min_plate_aspect_ratio, max_plate_aspect_ratio])

for combination in combs[start::6]:
  if combination is []: continue
  Main.run([0.1, 1.0], combination)