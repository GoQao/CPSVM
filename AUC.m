function [auc]=AUC(testlabel,predictlabel) 
x = 1.0;
y = 1.0;


[A,I]=sort(predictlabel); 
testlabel=testlabel(I);

M=0;N=0; 

for i=1:length(predictlabel) 
    if(testlabel(i)==1) 
        M=M+1;
    else 
        N=N+1;  
    end 
end 

x_step = 1.0/N;
y_step = 1.0/M;

for i=1:length(testlabel)
    if testlabel(i) == 1
        y = y - y_step;
    else
        x = x - x_step;
    end
    X(i)=x;
    Y(i)=y;
end

auc = -trapz(X,Y);

