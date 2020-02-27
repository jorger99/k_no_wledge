% plotting algorithm
clear

data_17 = load("laser_data_17.mat", '-mat', "laser_data_17")
data_10 = load("laser_data_10.mat", '-mat', "laser_data_10")


plot(data_17.laser_data_17(:,1), data_17.laser_data_17(:,2));
hold on
plot(data_10.laser_data_10(:,1), data_10.laser_data_10(:,2));

title('Photodiode Voltage vs. Position for Angle 0');
xlabel('Position (mm)');
ylabel('Voltage (V)');