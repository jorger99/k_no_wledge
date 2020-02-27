% this script will measure the laser profile
clear;
h=load_labjack;
counter = 0
mm_movement = 1.4;  % how many mm to move
conversion = 10000; % 10,000 steps per mm

laser_data_17 = getsomedata(mm_movement * conversion)
save("laser_data_17", "laser_data_17")

