#!/usr/bin/env python
import time
import sys
import os
import cv2
import numpy as np
from pynput import keyboard
import argparse

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot

parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--im_num', type = int, default = 0)
parser.add_argument('--folder', type = str, default = 'train')
args = parser.parse_args()

if not os.path.exists(script_path+"/../data/"+args.folder):
    data_path = script_path.replace('scripts', 'data')
    print(f'Folder "{args.folder}" in path {data_path} does not exist. Please create it.')
    exit()

bot = PiBot(ip=args.ip)
# stop the robot

bot.setVelocity(0, 0)

#countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")


# Initialize variables
angle = 0
im_number = args.im_num
continue_running = True
# Dead-man: robot moves only while one of Up/Left/Right is held. Track which keys are held.
drive_keys_held = set()

def on_press(key):
    global angle, continue_running, drive_keys_held
    try:
        if key == keyboard.Key.up:
            angle = 0
            print("straight")
            drive_keys_held.add(key)
        elif key == keyboard.Key.down:
            angle = 0
        elif key == keyboard.Key.right:
            print("right")
            angle += 0.1
            drive_keys_held.add(key)
        elif key == keyboard.Key.left:
            print("left")
            angle -= 0.1
            drive_keys_held.add(key)
        elif key == keyboard.Key.space:
            print("quit")
            bot.setVelocity(0, 0)
            continue_running = False

    except Exception as e:
        print(f"An error occurred: {e}")
        bot.setVelocity(0, 0)

def on_release(key):
    global drive_keys_held
    try:
        if key in (keyboard.Key.up, keyboard.Key.left, keyboard.Key.right):
            drive_keys_held.discard(key)
            if not drive_keys_held:
                bot.setVelocity(0, 0)
    except Exception as e:
        print(f"An error occurred: {e}")
        bot.setVelocity(0, 0)

# Start the listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Any of Up / Left / Right held => drive and record. Release all => stop (no record).
drive_keys = {keyboard.Key.up, keyboard.Key.left, keyboard.Key.right}

try:
    while continue_running:
        # Get an image from the robot
        img = bot.getImage()

        drive_active = bool(drive_keys_held & drive_keys)
        if not drive_active:
            bot.setVelocity(0, 0)
            time.sleep(0.05)
            continue

        angle = np.clip(angle, -0.5, 0.5)
        Kd = 25  # Base wheel speeds
        Ka = 25  # Turn speed
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)

        bot.setVelocity(left, right)

        cv2.imwrite(script_path+"/../data/"+args.folder+"/"+str(im_number).zfill(6)+'%.2f'%angle+".jpg", img)
        im_number += 1

        time.sleep(0.1)  # Small delay to reduce CPU usage

    # Clean up
    bot.setVelocity(0, 0)
    listener.stop()
    print("Script ended")


except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
    listener.stop()