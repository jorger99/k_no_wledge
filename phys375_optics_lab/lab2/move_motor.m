function data = move_motor(steps)

h=load_labjack;
i=0;
while (i<steps)
    lj_step(h)
    i=i+1;
end

clear;
