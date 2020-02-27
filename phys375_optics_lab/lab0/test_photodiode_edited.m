%connect photodetector to AIN0 on LabJack
%Press control-c to stop
clear;
h=load_labjack;

oldstr = num2str(lj_get(h));
while 1
    newstr = num2str(lj_get(h));
    fprintf([repmat('\b',1,length(oldstr)), '%s'],newstr );
    oldstr = newstr;
    pause(0.05)
end
