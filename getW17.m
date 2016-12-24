function [W, K, n1, n2]=getW17(Xs, Xt, mu, lambda, dim, kind, p1, p2, p3, labels1)
%ITCA方法
%Xs：源域数据
%Xt：目标域数据，与Xs行数相同
%mu：平衡因子，越大越重视映射后的相似度，越小越重视W的复杂度
%lambda：平衡因子，越大越重视映射后的同类数据的相似性，越小越重视W的复杂度
%dim：当dim为大于等于1的整数时，dim为降维的目标维数；
%     当dim为大于0小于1的小数时，所取特征向量对应的特征值的和>=全部特征值加和*dim
%kind：核函数选择:'Gaussian'、'Laplacian'、'Polynomial',其他一律返回-1
%p1,p2,p3：核函数所要附带的参数
%W：变换矩阵n1+n2->dim
%K：待变换矩阵
%n1,n2：源数据，目标数据的数目

    s1=size(Xs);
    s2=size(Xt);
    n1=s1(2);n2=s2(2);
    labels1 = labels1(1:n1);
    
%%%%%%%%%%% 计算K %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    X(:,1:n1)=Xs;
    X(:,n1+1:n1+n2)=Xt;
    for i=1:n1+n2 
        for j=i:n1+n2 
            K(i,j)=getKernel(X(:,i), X(:,j), kind, p1, p2, p3);
            K(j,i)=K(i,j);
        end
    end
    
%%%%%%%%%%% 计算L %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    L(1:n1, 1:n1)=ones(n1, n1)/(n1*n1);
    L(n1+1:n1+n2, n1+1:n1+n2)=ones(n2, n2)/(n2*n2);
    L(1:n1, n1+1:n1+n2)=ones(n1, n2)/(-n1*n2);
    L(n1+1:n1+n2, 1:n1)=ones(n2, n1)/(-n1*n2);
    
%%%%%%%%%%% 计算H %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H=eye(n1+n2)-ones(n1+n2, n1+n2)/(n1+n2);
    
%%%%%%%%%%% 计算类内距离 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%x-x
    
    t=zeros(n1+n2);
    ind=1:n1;
    count=0;
    for i=min(labels1):max(labels1)
        a=(labels1==i);
        s=sum(a);
        ind1=ind(a);
        t1=zeros(n1+n2);
        for j=1:s
            for k1=j+1:s
                t1(ind1(j),ind1(j))=t1(ind1(j),ind1(j))+1;
                t1(ind1(j),ind1(k1))=-1;
                t1(ind1(k1),ind1(j))=-1;
                t1(ind1(k1),ind1(k1))=t1(ind1(k1),ind1(k1))+1;
            end
        end
        if s>1
           t1=t1*2/s/(s-1); 
           count=count+1;
        end
        t=t+t1;
    end
    t=t/count;
    V1=K*t*K;    

%%%%%%%%%%% 计算W %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Temp=(1*eye(n1+n2)+mu*K*L*K+lambda*V1)^(-1)*(K*H*K);
    [V,D]=eig(Temp);
    D=diag(D);
    D=real(D);
    [D,I]=sort(D,'descend');
    
    if dim>0 && dim<1
        count=1;
        cur=0;
        s=sum(D);
        while cur/s<dim && D(count)>0
            cur=cur+D(count);
            count=count+1;
        end
    else
        count=dim+1;
    end
    
    I=I(1:count-1,1);
    W=V(:,I');

    
