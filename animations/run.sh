#!/bin/bash
# fps=$1
fps=${1:-60}
resolution=${2:-"1920,1080"}
# half_fps=$(($fps/2))

for scene in TrigScene ReluScene AbsScene; do
    nohup manim -r $resolution --fps=$fps main.py $scene 2>&1 > $scene.log &
done
