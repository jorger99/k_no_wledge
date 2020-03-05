% this script will measure the laser profile

function data = laser_polarization(steps)


laser_data = step_polarizer(steps)

dataloc = "data_partD/";
filename = "2cornsyrups_D6";

disp('Click on window if you are okay saving as: ' + filename);
k = waitforbuttonpress;

savepath = dataloc + filename + ".txt"
save(savepath, 'laser_data', '-ascii');
disp('Saved. End function.');

end