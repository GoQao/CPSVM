function Lh = Lhinge(A1,A2,B1,B2,w,wp,g1,g2)

A=[A1;A2];B=[B1;B2];
[im,~] = size(A1);
[in,~] = size(A2);

pos_train_label_predict = kernel(A1,A,'rbf',g1,1)*w;
neg_train_label_predict = kernel(A2,A,'rbf',g1,1)*w;
rho_pos = kernel(B1,B,'rbf',g2,1)*wp;
rho_neg = kernel(B2,B,'rbf',g2,1)*wp;

M = rho_pos*ones(im,1) - pos_train_label_predict;
N = rho_neg*ones(in,1) + neg_train_label_predict;
Z = [M;N];
Z(Z<0)=0;

Lh.Z = Z;
end