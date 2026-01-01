function model=train_ex_cpi(A1,A2,B1,B2,C,D,lambda,type,gamma1,gamma2)

[l1,~] = size(A1);
[l2,~] =size(A2);
A=[A1;A2];B=[B1;B2];
l=l1+l2;
ka1=kernel(A1,A,type,gamma1,1);ka2=kernel(A2,A,type,gamma1,1);
k=kernel(A,A,type,gamma1,1);
kb1=kernel(B1,B,type,gamma2,1);kb2=kernel(B2,B,type,gamma2,1);
kp=kernel(B,B,type,gamma2,1);

options = optimset;
options.LargeScale = 'off';
options.Display = 'off';

H=[k, zeros(l,l+2);
    zeros(2,l),lambda*kp,zeros(2,l);
    zeros(l,2*l+2)];
f=[zeros(l,1);-D*(kb1'+kb2');C];
A = [-ka1, ones(l1,1)*kb1,-eye(l1,l1),zeros(l1,l2);
   ka2, ones(l2,1)*kb2,zeros(l2,l1),-eye(l2,l2);
   zeros(1,l), -kb1,zeros(1,l);
   zeros(1,l),-kb2,zeros(1,l);
   zeros(l,l+2), -eye(l,l)];

b = [zeros(2*l+2,1)];


a0 = zeros(2*l+2,1);
options.Algorithm='interior-point-convex';
[a]  = quadprog(H,f,A,b,[],[],[],[],a0,options);
alpha=a(1:l);
beta=a(l+1:l+2);
ksi=a(l+3:end);
model.alpha=alpha;
model.beta=beta;
model.ksi=ksi;
end