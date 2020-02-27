% this script will measure the laser profile

function data = laser_profile(distance)

mm_movement = 19;  % how many mm to move
conversion = 10000; % 10,000 steps per mm

laser_data = collect_data(mm_movement * conversion)

filename = "data_partB/distance_" + distance + "mm.txt";
save(filename, 'laser_data', '-ascii');

end