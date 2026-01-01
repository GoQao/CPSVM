clear;clc;
for i=1:45
    name=num2str(i);
    filename1= strcat('C:\Users\GoQao\Desktop\类别特权学习\CGPL2\data\awa2\xy\',name,'.mat');
    filename2= strcat('C:\Users\GoQao\Desktop\类别特权学习\CGPL2\data\awa2\pi\',name,'.mat');
    f1 = load(filename1);
    f2 = load(filename2);
    X = f1.X;
    Y = f1.Y;
    p = f2.p;
    savefile=strcat(name,'.mat');
    save(savefile,'X','Y','p')

end