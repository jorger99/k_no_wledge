% this script will measure the laser profile over the entire x-axis span
% of the experiment and save the data 
mm_movement = 10;  % how many mm to move; 13.5 is the range of -70 to 70 deg


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%CHANGE VARIABLE EVERY EXPERIMENT%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

angle = "neg50"  % label data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%CHANGE VARIABLE EVERY EXPERIMENT%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

laser_data = getsomedata_b17((mm_movement * 10000), angle) % collect data using func

filename = "data/refraction_angle" + angle + ".txt"
save(filename, 'laser_data', '-ascii')

