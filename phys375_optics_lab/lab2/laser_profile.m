% this script will measure the laser profile

function data = laser_profile(distance, scan_mm)

%mm_movement = 5;  % how many mm to move
conversion = 10000; % 10,000 steps per mm

laser_data = collect_data(scan_mm * conversion)

filename = "data_partE/distance_" + distance + "mm.txt";
save(filename, 'laser_data', '-ascii');

end