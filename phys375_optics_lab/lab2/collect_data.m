function data = collect_data(steps)

h = load_labjack;
data = zeros(steps,2);

figure
for i=1:steps
    data(i,1) = i/10000.0; %x position in mm
    data(i,2) = lj_get(h); %voltage measurement in volts
    lj_step(h);
end

disp('Click on window when you have reversed the motor.');
plot(data(:,1),data(:,2));
title(('Photodiode Voltage vs. Position for Angle 0'));
xlabel('Position (mm)');
ylabel('Voltage (V)');
k = waitforbuttonpress;
for i=1:steps
    lj_step(h);
end