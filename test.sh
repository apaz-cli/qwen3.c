#!/bin/sh

rm Qwen3-0.6B.bin
./build.sh 0.6B
./qwen3-0.6B Qwen3-0.6B.bin \
  -i "Could you tell me a about a cat afraid of a dog?"
