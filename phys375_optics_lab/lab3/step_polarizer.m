function data = step_polarizer(num_steps)

h = load_labjack;
data = zeros(num_steps,2);


for i=1:num_steps
    lj_step(h);
    data(i,1)=i; % x position in steps
    data(i,2)=lj_get(h);  % y_val, voltage, in volts
end


plot(data(:,1), data(:,2))
