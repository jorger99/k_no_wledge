% this script will be used to take several zero angle measurements
mm_movement = 3;  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%CHANGE VARIABLE EVERY EXPERIMENT%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

datalabel = "air_1"  % label data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%CHANGE VARIABLE EVERY EXPERIMENT%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

laser_data = getsomedata_b17((mm_movement * 10000), datalabel) % collect data using func

filename = "data/zero_test_" + datalabel + ".txt"
save(filename, 'laser_data', '-ascii')

