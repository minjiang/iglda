function [Ys, Yt]=getY(W, K, n1, n2)
%对用于获取W的源域数据和目标域数据进行映射
%W：变换矩阵n1+n2->k
%K：待变换矩阵
%n1,n2：源数据，目标数据的数目
%Ys,Yt：变换后的矩阵

    Y=W'*K;
    Ys=Y(:,1:n1);
    Yt=Y(:,n1+1:n1+n2);
